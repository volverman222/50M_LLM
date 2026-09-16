"""Local content identity gate for reviewed ML source; no network operations."""
import hashlib
import subprocess
from pathlib import Path


def capture(repo):
    root=Path(repo).resolve()
    script=root/'scripts/train_pretrain_1b.py'
    if not script.is_file():raise ValueError('Training source entry point is missing')
    paths=[script,*sorted((root/'src').rglob('*.py'))]
    if (root/'pyproject.toml').is_file():paths.append(root/'pyproject.toml')
    files={}
    for path in paths:
        if not path.resolve().is_relative_to(root):raise ValueError('Training source escapes repository')
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    commit=None
    try:
        p=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,timeout=5)
        if p.returncode==0:commit=p.stdout.strip()
    except (OSError,subprocess.TimeoutExpired):pass
    return dict(root=str(root),git_sha=commit,files=files,
                scope='entry point, importable Python sources and project dependencies; excludes credentials/data')


def verify(identity):
    if not isinstance(identity,dict) or not identity.get('files'):
        raise ValueError('Training source identity is missing')
    actual=capture(identity['root'])
    if actual['files']!=identity['files'] or actual['git_sha']!=identity.get('git_sha'):
        raise ValueError('Training source changed after recipe preparation; review and recompile')
    return actual
