"""Source-only local releases. Never publishes, overwrites, or copies run data."""
from __future__ import annotations
import hashlib
import json
import shutil
import tempfile
from datetime import datetime,timezone
from pathlib import Path

MANIFEST='package_identity.json'
PATTERNS=('src/**/*.py','tests/**/*.py','adapters/**/*.py','docs/**/*.md','examples/*.json','examples/*.md')


def inventory(root):
    root=Path(root).resolve()
    files={root/'pyproject.toml',root/'README.md'}
    for pattern in PATTERNS:files.update(root.glob(pattern))
    if not (root/'src/exp_orchestrator/__init__.py').is_file():
        raise ValueError('Missing orchestrator package')
    result={}
    for path in sorted(files):
        if '__pycache__' in path.parts:continue
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Package file is missing or escapes source root')
        result[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def content_id(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def export_package(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Source and release must be separate trees')
    if destination.exists():raise FileExistsError(destination)
    files=inventory(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.release-',dir=destination.parent))
    try:
        for name,expected in files.items():
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,target)
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                raise ValueError('Source identity changed during copy')
        if inventory(source)!=files:raise ValueError('Source identity changed during export')
        receipt=dict(schema_version=1,content_sha256=content_id(files),files=files,
            created_at=datetime.now(timezone.utc).isoformat(),
            scope='source-only local snapshot; not signed or approved for publication')
        (stage/MANIFEST).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        verify_package(stage)
        if destination.exists():raise FileExistsError(destination)
        stage.rename(destination)
        return receipt
    finally:
        if stage.exists():shutil.rmtree(stage)


def verify_package(root):
    root=Path(root)
    try:
        receipt=json.loads((root/MANIFEST).read_text(encoding='utf-8'))
        actual=inventory(root)
        valid=(type(receipt.get('schema_version')) is int and receipt['schema_version']==1
            and receipt['files']==actual and receipt['content_sha256']==content_id(actual))
    except (OSError,KeyError,TypeError,ValueError) as exc:
        raise ValueError('Package identity cannot be verified') from exc
    if not valid:raise ValueError('Package identity mismatch')
    return receipt


def compare_package(source,release):
    expected=inventory(source);actual=verify_package(release)['files']
    changed=sorted(k for k in expected.keys() & actual.keys() if expected[k]!=actual[k])
    added=sorted(expected.keys()-actual.keys());removed=sorted(actual.keys()-expected.keys())
    return dict(matches=not(changed or added or removed),changed=changed,
        added_in_source=added,missing_from_source=removed,
        source_content_sha256=content_id(expected),release_content_sha256=content_id(actual))
