"""Local experiment management; publication is never a CLI operation."""
import argparse
import json
from pathlib import Path
from . import operations
from .locking import worker_lock


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('submit').add_argument('spec',type=Path)
    sub.add_parser('list');sub.add_parser('show').add_argument('run_id')
    log=sub.add_parser('logs');log.add_argument('run_id');log.add_argument('--stream',default='stdout',choices=['stdout','stderr','telemetry','resources','config','results','data-protocol','data-receipt','data-used','validation','source','events'])
    sub.add_parser('rebuild');sub.add_parser('worker').add_argument('--stay-alive',action='store_true');sub.add_parser('cancel').add_argument('run_id')
    serve=sub.add_parser('serve');serve.add_argument('--port',type=int,default=8766);serve.add_argument('--allow-control',action='store_true')
    watch=sub.add_parser('watch');watch.add_argument('--run-id');watch.add_argument('--stream',default='stdout');watch.add_argument('--once',action='store_true')
    prep=sub.add_parser('prepare-ml');prep.add_argument('recipe',type=Path);prep.add_argument('--output',type=Path)
    export=sub.add_parser('snapshot-package');export.add_argument('source',type=Path);export.add_argument('destination',type=Path)
    verify=sub.add_parser('verify-package');verify.add_argument('release',type=Path);verify.add_argument('--source',type=Path)
    sub.add_parser('worker-status')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=='submit':print(operations.submit(args.root,json.loads(args.spec.read_text(encoding='utf-8-sig'))))
    elif args.action=='prepare-ml':
        from .ml import build_spec, resolve_recipe_paths
        spec=build_spec(args.root,resolve_recipe_paths(json.loads(args.recipe.read_text(encoding='utf-8-sig')),args.recipe.parent))
        text=spec.model_dump_json(indent=2)
        if args.output:
            with args.output.open('x',encoding='utf-8') as f:f.write(text+'\n')
        else:print(text)
    elif args.action=='list':print(json.dumps(operations.snapshot(args.root),indent=2))
    elif args.action=='show':
        store,_,_=operations.services(args.root);print(json.dumps(store.load_run(args.run_id),indent=2))
    elif args.action=='logs':print(operations.read_tail(args.root,args.run_id,args.stream))
    elif args.action=='cancel':
        operations.request_cancel(args.root,args.run_id);print('Cancellation requested; worker confirmation pending.')
    elif args.action=='rebuild':
        with worker_lock(args.root):
            with worker_lock(args.root/'.admission',timeout=10):
                _,ledger,_=operations.services(args.root);ledger.rebuild()
        print('Ledger rebuilt from evidence.')
    elif args.action=='worker':return operations.run_worker(args.root,stay_alive=args.stay_alive)
    elif args.action=='worker-status':print(json.dumps(operations.worker_status(args.root),indent=2))
    elif args.action=='snapshot-package':
        from .distribution import export_package
        print(json.dumps(export_package(args.source,args.destination),indent=2))
    elif args.action=='verify-package':
        from .distribution import verify_package,compare_package
        result=compare_package(args.source,args.release) if args.source else verify_package(args.release)
        print(json.dumps(result,indent=2))
        if result.get('matches') is False:return 1
    elif args.action=='watch':
        from .monitor import monitor,terminal_view
        if args.once:
            from rich.console import Console
            Console().print(terminal_view(args.root,args.run_id,args.stream))
        else:return monitor(args.root,args.run_id,args.stream)
    elif args.action=='serve':
        from .web import make_server
        server=make_server(args.root,args.port,args.allow_control)
        print(f'Devpost Hackathon dashboard: http://127.0.0.1:{server.server_port}',flush=True)
        if args.allow_control:print(f'Control token (local session only): {server.control_token}',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
        finally:server.server_close()
    return 0


if __name__=='__main__':raise SystemExit(main())
