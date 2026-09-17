"""OS-owned locks; exclusive worker by default, bounded waits for admission."""
from contextlib import contextmanager
from pathlib import Path
import errno
import math
import os
import time


@contextmanager
def worker_lock(root,timeout=0):
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<0:
        raise ValueError('Lock timeout must be finite and nonnegative')
    path=Path(root)/'worker.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        deadline=time.monotonic()+timeout
        while True:
            stream.seek(0)
            try:
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK) or time.monotonic()>=deadline:raise
                time.sleep(min(.02,max(0,deadline-time.monotonic())))
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)
