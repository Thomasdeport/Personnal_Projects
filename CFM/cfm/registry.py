"""Feature blocks: write one function, get a cached, versioned, testable block.

    @block('ticks', family='ticks')
    def ticks(r):                      # r is a features.common.Raw over a chunk of windows
        return {'spread_1tick': np.nanmean(r.spread_t == 1, 1), ...}

kind='window' → dict of (n,) arrays     → stored (n, d) float32   (tabular models, neural context)
kind='event'  → dict of (n,100) arrays  → stored (n, 100, d) float32 (neural event channels)
kind='token'  → one (n,100) int array   → stored (n, 100) int16, with `cardinality` (neural embeddings)

The cache key hashes the function source, the shared helpers, ctx (tick, lot) and the raw cache.
Editing a block therefore recomputes only that block. No block may look at labels.
"""
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import io

BLOCKS = {}


@dataclass
class Block:
    name: str
    fn: object
    kind: str
    family: str
    cardinality: int = 0
    doc: str = ''


def block(name, kind='window', family='misc', cardinality=0):
    def wrap(fn):
        if name in BLOCKS:
            raise ValueError(f'duplicate block {name}')
        BLOCKS[name] = Block(name, fn, kind, family, cardinality, (fn.__doc__ or '').strip())
        return fn
    return wrap


def catalogue():
    from . import features  # noqa: F401  (registers every block)
    import pandas as pd
    return pd.DataFrame([{'block': b.name, 'kind': b.kind, 'family': b.family, 'doc': b.doc.split('\n')[0]}
                         for b in BLOCKS.values()])


def _key(b, lab_dir, ctx):
    from .features import common
    h = hashlib.sha256()
    h.update(inspect.getsource(b.fn).encode())
    h.update(inspect.getsource(common).encode())
    h.update(json.dumps(ctx, sort_keys=True).encode())
    h.update(json.dumps(io.meta(lab_dir)['source'], sort_keys=True).encode())
    return h.hexdigest()[:10]


def context(lab_dir, lot):
    m = io.meta(lab_dir)
    return {'tick': float(m['tick']), 'lot': float(lot), 'n_venue': len(m['venue_vocab'])}


def compute(name, lab_dir, split, lot=100., chunk=20000, verbose=True):
    """Return (array memmap, column names). Computes once per (block source, ctx, raw cache)."""
    from . import features  # noqa: F401
    from .features.common import Raw
    b = BLOCKS[name]
    ctx = context(lab_dir, lot)
    folder = Path(lab_dir) / 'features' / split
    folder.mkdir(parents=True, exist_ok=True)
    key = _key(b, lab_dir, ctx)
    path, side = folder / f'{name}-{key}.npy', folder / f'{name}-{key}.json'
    if path.exists() and side.exists():
        return np.load(path, mmap_mode='r'), json.loads(side.read_text())['names']
    raw = io.load(lab_dir, split)
    n = len(raw['ids'])
    out, names = None, None
    for s in range(0, n, chunk):
        r = Raw({k: raw[k][s:s + chunk] for k in ['num', 'cat', 'oid']}, ctx)
        res = b.fn(r)
        if b.kind == 'token':
            vals = np.asarray(res)
            if vals.min() < 0 or vals.max() >= b.cardinality:
                raise ValueError(f'{name}: token outside [0,{b.cardinality})')
            vals, cur_names = vals.astype(np.int16), [name]
        else:
            cur_names = list(res)
            vals = np.stack([np.asarray(res[k], dtype=np.float32) for k in cur_names], -1)
        if out is None:
            names = cur_names
            out = np.lib.format.open_memmap(path.with_suffix('.tmp.npy'), 'w+', vals.dtype, (n,) + vals.shape[1:])
        elif cur_names != names:
            raise ValueError(f'{name}: column names change between chunks')
        out[s:s + len(vals)] = vals
    out.flush(); del out
    path.with_suffix('.tmp.npy').replace(path)
    side.write_text(json.dumps({'names': names, 'kind': b.kind, 'family': b.family, 'ctx': ctx}))
    for stale in folder.glob(f'{name}-*.npy'):
        if stale != path and not stale.name.endswith('.tmp.npy'):
            stale.unlink(); stale.with_suffix('.json').unlink(missing_ok=True)
    if verbose:
        print(f'[block {name}/{split}] {len(names)} columns, key {key}', flush=True)
    return np.load(path, mmap_mode='r'), names


def block_key(name, lab_dir, lot=100.):
    from . import features  # noqa: F401
    return _key(BLOCKS[name], lab_dir, context(lab_dir, lot))


def window_matrix(blocks, lab_dir, split, lot=100.):
    """Concatenate window blocks → X (n, D) float32 in RAM, names, family per column."""
    from . import features  # noqa: F401
    xs, names, fams = [], [], []
    for b in blocks:
        if BLOCKS[b].kind != 'window':
            raise ValueError(f'{b} is a {BLOCKS[b].kind} block, not a window block')
        x, nm = compute(b, lab_dir, split, lot)
        xs.append(np.asarray(x)); names += [f'{b}:{c}' for c in nm]; fams += [BLOCKS[b].family] * len(nm)
    return np.concatenate(xs, 1), names, fams
