"""Resolve only the explicitly selected trusted model repository."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
from pathlib import Path


def load_protocol(repo):
    root=Path(repo).resolve()
    path=root/'src/llm_mini_lab/data_protocol.py'
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Selected model repository has no contained data protocol module')
    name='_devpost_protocol_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for required in ('validate','_keys','fingerprint','validate_runtime'):
        if not callable(getattr(module,required,None)):
            raise ValueError('Selected model protocol has an unsupported interface')
    return module


def resolve_recipe_paths(recipe,base_dir):
    result=copy.deepcopy(recipe);base=Path(base_dir).resolve()
    def resolve(value):
        path=Path(value).expanduser()
        return str((path if path.is_absolute() else base/path).resolve())
    for key in ('model_repo','tokenizer_model'):
        if result.get(key):result[key]=resolve(result[key])
    execution=result.get('execution',{})
    if execution.get('python_executable'):
        execution['python_executable']=resolve(execution['python_executable'])
    source=result.get('protocol',{}).get('source',{})
    if source.get('backend')=='local_jsonl':
        source['files']={role:resolve(path) for role,path in source['files'].items()}
    return result
