"""Loopback-only dashboard. Read-only unless explicitly enabled with a token.

This is an operator interface to trusted local processes, not a sandbox or a
multi-user remote service. Do not expose it via a public reverse proxy.
"""
import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,unquote
from .operations import snapshot,submit,read_tail,request_cancel,worker_status

PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Devpost Hackathon — Experiment orchestrator</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px system-ui;margin:2rem auto;max-width:1100px;padding:0 1rem}h1{text-align:center}table{width:100%;border-collapse:collapse}td,th{padding:.6rem;border-bottom:1px solid #bbb;text-align:left}button,select,input{padding:.5rem;margin:.2rem}pre,textarea{box-sizing:border-box;width:100%;white-space:pre-wrap;overflow-wrap:anywhere;max-height:32rem;overflow:auto;border:1px solid #bbb;padding:1rem}textarea{min-height:10rem}.muted{font-size:.9rem}#notice{white-space:pre-wrap}</style>
<h1>Devpost Hackathon — 50M_LLM</h1><h2>Experiment queue and evidence</h2>
<p class="muted">Local operator interface. Controls require an explicitly enabled server and its session token. Reads refresh every 2 seconds. Evidence views are bounded to 16 KiB.</p>
<p id="worker-state" role="status">Checking worker lifecycle...</p><table><thead><tr><th>Run</th><th>Status</th><th>Profile</th><th>Priority</th></tr></thead><tbody id="runs"></tbody></table>
<p><label>Selected run <select id="selected"></select></label></p>
<div id="tabs"></div><h3 id="heading">Logs — stdout</h3><pre id="output">Select a run.</pre>
<details><summary>Queue / cancel a reviewed local command</summary>
<p>Submitting here authorizes local execution by the worker. This does not publish code or upload models.</p>
<label>Control token <input id="token" type="password" autocomplete="off"></label>
<textarea id="spec" aria-label="Run specification JSON" placeholder="Paste a reviewed RunSpec JSON"></textarea>
<button id="submit">Queue specification</button><button id="cancel">Request cancellation</button></details>
<p id="notice" role="status"></p>
<script>
const $=id=>document.getElementById(id);let stream='stdout';
const tabs={stdout:'Logs',stderr:'Errors',telemetry:'Telemetry',resources:'Resources',config:'Config',results:'Results','data-protocol':'Data / splits','data-used':'Inputs consumed',validation:'Validation identity',source:'Source identity',events:'Events'};
for(const [key,label] of Object.entries(tabs)){const b=document.createElement('button');b.textContent=label;b.onclick=()=>{stream=key;refreshOutput()};$('tabs').appendChild(b)}
async function refreshOutput(){let id=$('selected').value;if(!id)return;$('heading').textContent=tabs[stream]+' — '+id;const r=await fetch('/api/run/'+encodeURIComponent(id)+'/'+stream);$('output').textContent=await r.text()||'No evidence yet.'}
async function refresh(){try{const w=await fetch('/api/worker');if(w.ok){const state=await w.json();$('worker-state').textContent='Worker: '+state.state+' | process alive: '+Boolean(state.alive)}const r=await fetch('/api/runs');if(!r.ok)throw Error(await r.text());const rows=await r.json();const old=$('selected').value;$('runs').replaceChildren();$('selected').replaceChildren();for(const row of rows){const tr=document.createElement('tr');for(const key of ['run_id','status','profile','priority']){const td=document.createElement('td');td.textContent=String(row[key]??'');tr.appendChild(td)}$('runs').appendChild(tr);const option=document.createElement('option');option.value=row.run_id;option.textContent=row.run_id;$('selected').appendChild(option)}if(rows.some(x=>x.run_id===old))$('selected').value=old;await refreshOutput()}catch(e){$('notice').textContent=String(e)}}
async function control(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':$('token').value},body:JSON.stringify(body)});$('notice').textContent=await r.text();await refresh()}
$('selected').onchange=refreshOutput;$('submit').onclick=()=>{try{control('/api/submit',JSON.parse($('spec').value))}catch(e){$('notice').textContent=String(e)}};$('cancel').onclick=()=>{const id=$('selected').value;if(id&&confirm('Request cancellation of '+id+'?'))control('/api/run/'+encodeURIComponent(id)+'/cancel',{})};refresh();setInterval(refresh,2000);
</script></html>"""


def make_server(root,port=0,allow_control=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass  # No credential tokens or arbitrary inputs in console logs.
        def safe_origin(self):
            hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            origin=self.headers.get('Origin')
            return self.headers.get('Host') in hosts and (origin is None or origin in {'http://'+h for h in hosts})
        def reply(self,status,body,ctype='text/plain; charset=utf-8'):
            raw=body.encode() if isinstance(body,str) else body
            self.send_response(status);self.send_header('Content-Type',ctype)
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self.safe_origin():return self.reply(403,'Origin/Host rejected')
            path=unquote(urlsplit(self.path).path)
            try:
                if path=='/':return self.reply(200,PAGE,'text/html; charset=utf-8')
                if path=='/api/worker':return self.reply(200,json.dumps(worker_status(root)),'application/json')
                if path=='/api/runs':return self.reply(200,json.dumps(snapshot(root)),'application/json')
                parts=path.split('/')
                if len(parts)==5 and parts[1:3]==['api','run']:
                    return self.reply(200,read_tail(root,parts[3],parts[4]))
                return self.reply(404,'Not found')
            except (ValueError,KeyError,FileNotFoundError):self.reply(400,'Invalid run or evidence path')
        def do_POST(self):
            if not self.safe_origin() or not self.server.allow_control or not hmac.compare_digest(self.headers.get('X-Control-Token',''),self.server.control_token):
                return self.reply(403,'Explicit control mode and token required')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=131072 or self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    return self.reply(400,'Expected JSON body, at most 128 KiB')
                data=json.loads(self.rfile.read(length));path=unquote(urlsplit(self.path).path)
                if path=='/api/submit':return self.reply(201,json.dumps({'run_id':submit(root,data)}),'application/json')
                parts=path.split('/')
                if len(parts)==5 and parts[1:3]==['api','run'] and parts[4]=='cancel':
                    request_cancel(root,parts[3]);return self.reply(202,'Cancellation requested; worker confirmation pending')
                self.reply(404,'Not found')
            except (ValueError,KeyError,OSError) as exc:self.reply(400,str(exc))
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True;server.allow_control=bool(allow_control);server.control_token=secrets.token_urlsafe(32)
    return server
