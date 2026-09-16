"""Read-only host/process sampling. Never mutates or terminates workloads."""
import csv
import io
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import psutil


def sample(pids, disk):
    result = dict(at=datetime.now(timezone.utc).isoformat(), monotonic=time.monotonic(),
        cpu_percent=psutil.cpu_percent(), system_ram=psutil.virtual_memory()._asdict(),
        disk_free_bytes=shutil.disk_usage(disk).free, processes=[], gpus=[])
    for pid in pids:
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
            for p in processes:
                try:
                    result['processes'].append(dict(pid=p.pid, name=p.name(),
                        rss_bytes=p.memory_info().rss, cpu=p.cpu_times()._asdict(),
                        io=p.io_counters()._asdict(), threads=p.num_threads()))
                except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    fields = ['uuid','memory.total','memory.used','memory.free','utilization.gpu','utilization.memory','temperature.gpu','power.draw']
    try:
        p = subprocess.run(['nvidia-smi','--query-gpu='+','.join(fields),
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        if p.returncode: raise RuntimeError(p.stderr.strip())
        for row in csv.reader(io.StringIO(p.stdout)):
            values = [x.strip() for x in row]
            def number(value):
                try: return float(value)
                except ValueError: return None
            result['gpus'].append(dict(uuid=values[0],**{
                name:number(value) for name,value in zip(fields[1:],values[1:])}))
    except (OSError,RuntimeError,subprocess.TimeoutExpired) as exc:
        result['gpu_sample_error'] = str(exc)
    return result


def append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a',encoding='utf-8') as file:
        file.write(json.dumps(record,allow_nan=False)+'\n')
