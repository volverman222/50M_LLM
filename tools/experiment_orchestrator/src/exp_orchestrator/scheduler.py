"""Single-owner, serial scheduler. Unknown running jobs block new launches."""
from __future__ import annotations
import json
import math
import shutil
import uuid
from pathlib import Path
from .domain import RunEvent, RunResult, RunSpec, RunState
from .evidence import TERMINAL_STATES, _atomic_json


class Scheduler:
    def __init__(self, evidence, ledger, agents):
        self.evidence, self.ledger = evidence, ledger
        self.agents = list(agents)
        self.active = {}

    def enqueue(self, spec: RunSpec):
        if spec.metadata.get("preview_only"):
            raise ValueError("preview specs cannot execute")
        if spec.run_id in spec.depends_on:
            raise ValueError('a run cannot depend on itself')
        for dep in spec.depends_on:
            if self.ledger.get_run(dep) is None:
                raise ValueError(f'unknown predecessor: {dep}')
        self.evidence.create_run(spec)
        self.ledger.project_run(spec.run_id)
        return spec.run_id

    def _transition(self, run_id, state, detail=None):
        row = self.ledger.get_run(run_id)
        self.evidence.append_event(run_id, RunEvent.transition(
            row['status'], state, actor='scheduler', detail=detail))
        self.ledger.project_run(run_id)

    def _finish(self, run_id, state, exit_code=None, message=None, metrics=None):
        before = self.ledger.get_run(run_id)["status"]
        self.evidence.seal_result(run_id, RunResult(run_id=run_id,
            status=RunState(state), exit_code=exit_code, message=message,
            metrics=metrics or {}))
        self.evidence.append_event(run_id, RunEvent.transition(
            before, state, actor="scheduler", detail={"reason": message}))
        self.ledger.project_run(run_id)

    def _completion(self, run_id, handle, result):
        if result.cancelled:
            return 'cancelled', 'cancelled by operator', {}
        if result.timed_out or result.exit_code != 0:
            return 'failed', 'timeout' if result.timed_out else 'nonzero exit', {}
        rule = handle.prepared.spec.metadata.get('completion', {})
        if not rule:
            return 'completed', 'process exited successfully', {}
        try:
            path = handle.prepared.run_dir / rule['summary']
            if not path.resolve().is_relative_to(handle.prepared.run_dir.resolve()):
                raise ValueError('completion summary escapes run directory')
            data = json.loads(path.read_text(encoding='utf-8'))
            for key in ('tokens_seen', 'updates', 'training_time_seconds'):
                value = data.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise ValueError('completion metrics must be finite nonnegative numbers')
            if data.get('target_reached') is not True or data['tokens_seen'] < rule['min_tokens']:
                return 'failed', 'training target not reached', {}
            return 'completed', 'training target verified', {
                k: float(data[k]) for k in ('tokens_seen', 'updates',
                'training_time_seconds') if k in data}
        except (OSError, KeyError, TypeError, ValueError) as exc:
            return 'failed', f'missing/invalid completion evidence: {exc}', {}

    def tick(self):
        for run_id, (executor, handle) in list(self.active.items()):
            observation = executor.observe(handle)
            if not observation.running:
                result = executor.collect(handle, self.evidence.run_dir(run_id))
                state, message, metrics = self._completion(run_id, handle, result)
                self._finish(run_id, state, result.exit_code, message, metrics)
                del self.active[run_id]
        if self.active:
            return
        # Never duplicate jobs after a controller crash: reconcile explicitly.
        if any(r['status'] in ('running', 'admitted') for r in self.ledger.query_runs()):
            return
        for row in self.ledger.query_runs({'status': 'queued'}):
            run_id = row['run_id']
            spec = RunSpec.model_validate(self.evidence.load_run(run_id)['manifest'])
            deps = [self.ledger.get_run(d) for d in spec.depends_on]
            if any(d and d['status'] in ('failed', 'cancelled', 'blocked') for d in deps):
                self._finish(run_id, 'blocked', message='predecessor did not succeed')
                continue
            if any(not d or d['status'] != 'completed' for d in deps):
                continue
            if self._launch(spec):
                return

    def _launch(self, spec):
        free_mb = shutil.disk_usage(self.evidence.run_dir(spec.run_id)).free / (1024**2)
        if free_mb < spec.resources.min_disk_free_mb:
            self._finish(spec.run_id, "blocked", message=f"insufficient disk: {free_mb:.0f} MiB free")
            return False
        compatible = False
        for agent in self.agents:
            caps = agent.capabilities()
            if spec.executor not in caps['executors']:
                continue
            req = spec.resources
            candidates = [g for g in caps.get('gpus', [])
                if g['memory_total_mb'] >= req.min_vram_mb]
            if len(candidates) < req.gpu_count:
                continue
            compatible = True
            free = [g for g in candidates if g['memory_free_mb'] >= req.min_vram_mb]
            if len(free) < req.gpu_count:
                continue
            executor = agent.executor(spec.executor)
            folder = self.evidence.run_dir(spec.run_id)
            self._transition(spec.run_id, 'admitted', {'host': caps['host_id']})
            _atomic_json(folder/'hardware.json', caps)
            try:
                prepared = executor.prepare(spec, folder)
                if req.gpu_count:
                    prepared.env['CUDA_VISIBLE_DEVICES'] = ','.join(
                        str(g.get('uuid', g['index'])) for g in free[:req.gpu_count])
                handle = executor.start(prepared)
                self.active[spec.run_id] = (executor, handle)
                self._transition(spec.run_id, 'running', {
                    'pid': handle.process.pid, 'host': caps['host_id']})
                return True
            except Exception as exc:
                self._finish(spec.run_id, 'failed', message=f'launch error: {exc}')
                return False
        if not compatible:
            self._finish(spec.run_id, 'blocked', message='no compatible host/resources')
        return False

    def cancel(self, run_id):
        row = self.ledger.get_run(run_id)
        if row is None:
            raise KeyError(run_id)
        if row['status'] in {s.value for s in TERMINAL_STATES}:
            return
        if run_id in self.active:
            executor, handle = self.active.pop(run_id)
            executor.stop(handle)
            result = executor.collect(handle, self.evidence.run_dir(run_id))
            self._finish(run_id, 'cancelled', result.exit_code, 'operator requested stop')
        elif row['status'] == 'queued':
            self._finish(run_id, 'cancelled', message='cancelled before launch')
        else:
            raise RuntimeError('cannot stop an unowned process; explicit recovery required')

    def retry(self, run_id):
        loaded = self.evidence.load_run(run_id)
        if not loaded['result']:
            raise ValueError('only terminal runs can be retried')
        if loaded['manifest']['profile'] != 'generic':
            raise ValueError('ML retries require a fresh explicitly approved manifest')
        data = loaded['manifest']
        data['run_id'] = f'{run_id[:60]}-retry-{uuid.uuid4().hex[:8]}'
        data['metadata']['retry_of'] = run_id
        from .domain import utc_now
        data['created_at'] = utc_now()
        return self.enqueue(RunSpec.model_validate(data))

    def set_priority(self, run_id, priority):
        row = self.ledger.get_run(run_id)
        if not row or row['status'] != 'queued':
            raise ValueError('only queued runs can change priority')
        self.evidence.append_event(run_id, RunEvent(kind='priority',
            actor='operator', detail={'priority': int(priority)}))
        self.ledger.project_run(run_id)
