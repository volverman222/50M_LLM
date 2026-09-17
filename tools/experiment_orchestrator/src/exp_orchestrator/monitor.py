"""Terminal monitoring client using the same services as the browser."""
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.live import Live
from .operations import snapshot,read_tail,worker_status


def terminal_view(root,run_id=None,stream='stdout'):
    table=Table(title='Devpost Hackathon — Experiment monitor')
    for name in ('Run','Status','Profile','Priority'):table.add_column(name)
    for row in snapshot(root):
        table.add_row(Text(row['run_id']),Text(row['status']),Text(row.get('profile','')),str(row.get('priority',0)))
    status=worker_status(root)
    panels=[Text('Worker: '+status['state']+' | process alive: '+str(status['alive'])),table]
    if run_id:panels.append(Panel(Text(read_tail(root,run_id,stream)),title=Text(f'{run_id} / {stream}')))
    return Group(*panels)


def monitor(root,run_id=None,stream='stdout',interval=2):
    if interval<=0:raise ValueError('interval must be positive')
    try:
        with Live(terminal_view(root,run_id,stream),console=Console(),refresh_per_second=1) as live:
            while True:
                time.sleep(interval);live.update(terminal_view(root,run_id,stream))
    except KeyboardInterrupt:return 0
