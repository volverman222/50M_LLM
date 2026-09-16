import http.client
import importlib.util
import json
import threading
from urllib.parse import urlsplit
import pytest


def setup_server(tmp_path):
    assert importlib.util.find_spec('exp_orchestrator'), 'Portable orchestrator package is missing'
    from exp_orchestrator.operations import submit
    from exp_orchestrator.web import make_server
    import sys
    submit(tmp_path, dict(run_id='demo', command=dict(argv=[sys.executable, '-c', 'print(1)'])))
    (tmp_path/'runs/demo/stdout.log').write_text('<script>alert(1)</script>')
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    return server, thread


def request(server, path, headers=None, method='GET', body=None):
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse(); data = response.read(); status = response.status
    headers = dict(response.getheaders()); conn.close()
    return status, data, headers


def test_dashboard_links_and_logs_are_read_only(tmp_path):
    server,thread = setup_server(tmp_path)
    try:
        status,body,headers = request(server, '/')
        assert status == 200 and b'Devpost Hackathon' in body and b'Logs' in body
        status,body,headers = request(server, '/api/runs')
        assert status == 200 and json.loads(body)[0]['run_id'] == 'demo'
        status,body,headers = request(server, '/api/run/demo/stdout')
        assert status == 200 and b'<script>' in body
        assert headers['Content-Type'].startswith('text/plain')
        assert headers['X-Content-Type-Options'] == 'nosniff'
        assert request(server, '/api/run/demo/../../secret')[0] in (400,404)
        assert request(server, '/api/run/demo/cancel', method='POST',body='{}')[0] == 403
    finally:
        server.shutdown(); server.server_close(); thread.join(5)


def test_dashboard_rejects_remote_origin_and_host(tmp_path):
    server,thread = setup_server(tmp_path)
    try:
        assert request(server, '/api/runs', {'Host':'evil.example'})[0] == 403
        assert request(server, '/api/runs', {'Origin':'https://evil.example'})[0] == 403
        assert request(server, '/', {'Origin':'null'})[0] == 403
    finally:
        server.shutdown(); server.server_close(); thread.join(5)


def test_control_requires_explicit_switch_and_token(tmp_path):
    assert importlib.util.find_spec('exp_orchestrator'), 'Portable orchestrator package is missing'
    from exp_orchestrator.web import make_server
    from exp_orchestrator.operations import snapshot
    import sys
    server = make_server(tmp_path, port=0, allow_control=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        data = json.dumps(dict(run_id='queued', command=dict(argv=[sys.executable,'-c','pass'])))
        assert request(server, '/api/submit', method='POST', body=data)[0] == 403
        headers = {'X-Control-Token':server.control_token, 'Content-Type':'application/json'}
        assert request(server,'/api/submit', headers, 'POST',data)[0] == 201
        assert snapshot(tmp_path)[0]['status']=='queued'
        assert request(server,'/api/run/queued/cancel', headers, 'POST','{}')[0] == 202
    finally:
        server.shutdown(); server.server_close(); thread.join(5)


def test_dashboard_reports_worker_lifecycle_separately_from_job_state(tmp_path):
    server,thread=setup_server(tmp_path)
    try:
        status,body,_=request(server,'/api/worker')
        assert status==200
        assert json.loads(body)['alive'] is False
        status,page,_=request(server,'/')
        assert b'worker-state' in page and b'/api/worker' in page
        assert b'Inputs consumed' in page and b'Validation identity' in page
    finally:
        server.shutdown();server.server_close();thread.join(5)
