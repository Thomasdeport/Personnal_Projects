"""JSON configs with `extends` and dotted overrides. Every run saves its resolved config."""
import copy
import hashlib
import json
from pathlib import Path


def deep_merge(base, update):
    out = copy.deepcopy(base)
    for k, v in update.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load(path, overrides=()):
    path = Path(path)
    cfg = json.loads(path.read_text())
    parent = cfg.pop('extends', None)
    if parent:
        cfg = deep_merge(load(path.parent / parent), cfg)
    for item in overrides:
        key, _, raw = item.partition('=')
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        *head, last = key.split('.')
        for h in head:
            node = node.setdefault(h, {})
        node[last] = value
    return cfg


def digest(obj, n=12):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def code_digest(*folders):
    h = hashlib.sha256()
    for folder in folders:
        for p in sorted(Path(folder).rglob('*.py')):
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]
