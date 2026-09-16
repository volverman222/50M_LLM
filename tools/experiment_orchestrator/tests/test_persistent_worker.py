import inspect
import json
import sys
import threading
import time
from exp_orchestrator import operations


def test_idle_worker_accepts_later_submission_and_reports_clean_stop(tmp_path):
    assert 'stay_alive' in inspect.signature(operations.run_worker).parameters
    stop=threading.Event();errors=[]
    def worker():
        try:operations.run_worker(tmp_path,interval=.02,resource_interval=.05,stay_alive=True,stop_event=stop)
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    try:
        end=time.monotonic()+5
        while not (tmp_path/'worker_status.json').exists() and time.monotonic()<end:time.sleep(.02)
        assert thread.is_alive() and not errors
        operations.submit(tmp_path,{'run_id':'late','command':{'argv':[sys.executable,'-c','print("late job")']}})
        end=time.monotonic()+10
        while not (tmp_path/'runs/late/result.json').exists() and time.monotonic()<end:time.sleep(.02)
        assert not errors, repr(errors)
        assert json.loads((tmp_path/'runs/late/result.json').read_text())['status']=='completed'
        assert 'late job' in operations.read_tail(tmp_path,'late','stdout')
        assert thread.is_alive()
    finally:
        stop.set();thread.join(10)
    assert not thread.is_alive() and not errors
    assert operations.worker_status(tmp_path)['state']=='stopped'


def test_worker_status_does_not_treat_old_receipt_as_a_live_process(tmp_path):
    assert hasattr(operations,'worker_status'), 'Worker lifecycle status missing'
    assert operations.worker_status(tmp_path)['alive'] is False
    (tmp_path/'worker_status.json').write_text(json.dumps({'pid':999999999,
        'process_created':0,'state':'running','heartbeat_at':'2000-01-01T00:00:00Z'}))
    result=operations.worker_status(tmp_path)
    assert result['alive'] is False and result['state']=='stale'


def test_worker_cli_stay_alive_is_explicit():
    from exp_orchestrator.cli import parser
    assert parser().parse_args(['--root','test','worker']).stay_alive is False
    assert parser().parse_args(['--root','test','worker','--stay-alive']).stay_alive is True


def test_admission_lock_waits_for_short_contention_without_masking_second_worker(tmp_path):
    from exp_orchestrator.locking import worker_lock
    assert 'timeout' in inspect.signature(worker_lock).parameters
    entered=threading.Event();release=threading.Event()
    def owner():
        with worker_lock(tmp_path):entered.set();release.wait(2)
    thread=threading.Thread(target=owner,daemon=True);thread.start()
    assert entered.wait(2)
    try:
        import pytest
        with pytest.raises(OSError):
            with worker_lock(tmp_path):pass
        timer=threading.Timer(.05,release.set);timer.start()
        with worker_lock(tmp_path,timeout=2):pass
        timer.join()
    finally:release.set();thread.join(3)
