"""Single-owner worker lifecycle; optional idle waiting, no implicit retries."""
import json
import os
import time
from datetime import datetime,timezone
from pathlib import Path
import psutil
from .evidence import _atomic_json
from .locking import worker_lock
from .resources import sample,append


def worker_status(root):
    path=Path(root)/'worker_status.json'
    if not path.exists():return {'state':'not_running','alive':False}
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        alive=(data['state'] not in ('stopped','failed','blocked')
            and abs(psutil.Process(data['pid']).create_time()-data['process_created'])<.01)
    except (OSError,ValueError,KeyError,TypeError,psutil.Error):
        return {'state':'stale','alive':False}
    data['alive']=bool(alive)
    if not alive and data['state'] not in ('stopped','failed','blocked'):data['state']='stale'
    return data


def run(root,interval=.1,resource_interval=2.,*,stay_alive=False,stop_event=None):
    from .operations import services
    if interval<=0 or resource_interval<=0 or type(stay_alive) is not bool:
        raise ValueError('Intervals must be positive and stay_alive boolean')
    root=Path(root).resolve();store,ledger,scheduler=services(root)
    last_resource=last_heartbeat=0.;final_state='stopped'
    identity={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'stay_alive':stay_alive}
    def publish(state):
        _atomic_json(root/'worker_status.json',dict(identity,state=state,
            heartbeat_at=datetime.now(timezone.utc).isoformat(),active_runs=list(scheduler.active)))
    with worker_lock(root):
        try:
            with worker_lock(root/'.admission',timeout=10):ledger.rebuild()
            publish('idle')
            while True:
                if stop_event is not None and stop_event.is_set():
                    for rid in list(scheduler.active):scheduler.cancel(rid)
                    return 0
                with worker_lock(root/'.admission',timeout=10):
                    for row in ledger.query_runs():
                        rid=row['run_id']
                        if (store.run_dir(rid)/'cancel.request').exists() and (rid in scheduler.active or row['status']=='queued'):
                            scheduler.cancel(rid)
                    scheduler.tick()
                if scheduler.active and time.monotonic()-last_resource>=resource_interval:
                    active={r:h.process.pid for r,(_,h) in scheduler.active.items()}
                    record=sample(list(active.values()),root)
                    for rid in active:append(store.run_dir(rid)/'telemetry/resources.jsonl',record)
                    last_resource=time.monotonic()
                if time.monotonic()-last_heartbeat>=1.:
                    publish('running' if scheduler.active else 'idle')
                    last_heartbeat=time.monotonic()
                if not scheduler.active:
                    rows=ledger.query_runs()
                    if any(r['status'] in ('running','admitted') for r in rows):
                        final_state='blocked';return 2
                    if not stay_alive and not any(r['status']=='queued' for r in rows):return 0
                if stop_event is not None:stop_event.wait(interval)
                else:time.sleep(interval)
        except KeyboardInterrupt:
            for rid in list(scheduler.active):scheduler.cancel(rid)
            return 130
        except BaseException:
            final_state='failed'
            for rid in list(scheduler.active):scheduler.cancel(rid)
            raise
        finally:publish(final_state)
