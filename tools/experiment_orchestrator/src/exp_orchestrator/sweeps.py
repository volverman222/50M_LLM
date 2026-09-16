"""Pure spec expansion. This module does not enqueue or execute experiments."""
import hashlib
import itertools
import json
import math
from .domain import RunSpec


def expand_sweep(base: RunSpec, grid: dict, campaign_id: str):
    keys = sorted(grid)
    if not keys or any(not grid[key] for key in keys):
        raise ValueError('grid must be nonempty')
    if math.prod(len(grid[key]) for key in keys) > 128:
        raise ValueError('explicit expansion limit is 128; nothing launched')
    children = []
    for values in itertools.product(*(grid[key] for key in keys)):
        data = base.model_dump(mode='json')
        changes = dict(zip(keys, values))
        for key, value in changes.items():
            parts = key.split('.')
            if parts[0] != 'metadata':
                raise ValueError('only metadata preview changes are supported')
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        digest = hashlib.sha256(json.dumps(changes, sort_keys=True).encode()).hexdigest()[:12]
        data['run_id'] = f'{campaign_id[:60]}-{digest}'
        data['campaign_id'] = campaign_id
        data['metadata']['sweep_changes'] = changes
        data['metadata']['execution_authorized'] = False
        data['metadata']['preview_only'] = True
        children.append(RunSpec.model_validate(data))
    if len({item.run_id for item in children}) != len(children):
        raise ValueError('duplicate grid values produce duplicate child IDs')
    return children
