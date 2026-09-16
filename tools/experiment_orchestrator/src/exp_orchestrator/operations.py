"""Shared local services for CLI/web/terminal clients; no implicit upload."""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path
from .domain import RunSpec
from .evidence import EvidenceStore, _atomic_json
from .ledger import Ledger
from .scheduler import Scheduler
from .agents.local import LocalHostAgent
from .executors.native import NativeExecutor
from .locking import worker_lock
from .resources import sample, append

TERMINAL = {'completed','failed','cancelled','blocked'}


def services(root):
    root=Path(root).resolve()
    store=EvidenceStore(root/'runs');ledger=Ledger(root/'ledger.db',store)
    scheduler=Scheduler(store,ledger,[LocalHostAgent({'native':NativeExecutor()})])
    return store,ledger,scheduler


def _no_secrets(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if re.search(r'(^|_)(api_key|password|secret|access_token|auth_token|credential)s?$',str(key),re.I):
                raise ValueError('Credential values must not be persisted in run specifications')
            _no_secrets(item)
    elif isinstance(value,list):
        for item in value:_no_secrets(item)
    elif isinstance(value,str) and re.match(r'^--(api-key|password|token|secret)(=|$)',value,re.I):
        raise ValueError('Credential command arguments are not permitted')


def submit(root,payload):
    payload=copy.deepcopy(payload);_no_secrets(payload)
    spec=RunSpec.model_validate(payload)
    if spec.max_retries:raise ValueError('Automatic retries are not supported; submit a reviewed new run')
    if spec.resources.process_limit!=1:raise ValueError('This worker is serial; process_limit must be 1')
    store,ledger,scheduler=services(root)
    store.run_dir(spec.run_id)  # Validate even before creating artifacts.
    with worker_lock(Path(root)/'.admission',timeout=10):
        if spec.run_id in spec.depends_on or any(ledger.get_run(d) is None for d in spec.depends_on):
            raise ValueError('Dependencies must refer to existing distinct predecessor runs')
        if spec.metadata.get('preview_only'):raise ValueError('Preview spec is not executable')
        folder=store.create_run(spec)
        _atomic_json(folder/'environment.json',dict(python_version=sys.version,executable=sys.executable,
            platform=platform.platform(),explicit_environment=spec.command.env))
        if 'source_provenance' in spec.metadata:
            _atomic_json(folder/'source.json',spec.metadata['source_provenance'])
        if 'data_protocol' in spec.metadata:
            _atomic_json(folder/'artifacts/data_protocol.json',spec.metadata['data_protocol'])
        ledger.project_run(spec.run_id)
    return spec.run_id


def snapshot(root):
    _,ledger,_=services(root)
    return ledger.query_runs()


def read_tail(root,run_id,stream,limit=16384):
    mapping={'stdout':'stdout.log','stderr':'stderr.log','telemetry':'telemetry/training.jsonl',
        'resources':'telemetry/resources.jsonl','config':'manifest.json','results':'result.json',
        'data-receipt':'checkpoints/data_protocol_receipt.json','data-used':'checkpoints/training_input_identity.json','validation':'checkpoints/fixed_validation_identity.json','source':'source.json',
        'data-protocol':'artifacts/data_protocol.json','events':'events.jsonl'}
    if stream not in mapping:raise ValueError('Unknown evidence stream')
    if not isinstance(limit,int) or not 1<=limit<=1048576:raise ValueError('Invalid read limit')
    store,_,_=services(root);folder=store.run_dir(run_id);path=folder/mapping[stream]
    if not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Evidence path escapes run')
    if not path.is_file():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit));raw=f.read(limit)
    return raw.decode('utf-8',errors='ignore')


def request_cancel(root,run_id):
    store,_,_=services(root);store.load_run(run_id)
    (store.run_dir(run_id)/'cancel.request').touch(exist_ok=True)


def worker_status(root):
    from .worker import worker_status as status
    return status(root)


def run_worker(root,interval=0.1,resource_interval=2.0,*,stay_alive=False,stop_event=None):
    from .worker import run
    return run(root,interval,resource_interval,stay_alive=stay_alive,stop_event=stop_event)
