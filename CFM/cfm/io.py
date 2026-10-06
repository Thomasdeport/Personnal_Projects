"""CSV → raw cache. No feature engineering here: only validated, aligned, typed arrays.

Layout of <lab>/raw/:
    {split}_num.npy  float32 (n,100,6)  price, bid, ask, bid_size, ask_size, flux — raw values, nothing clipped
    {split}_cat.npy  int8    (n,100,4)  venue, action, side, trade — 0 = unknown/missing
    {split}_oid.npy  int8    (n,100)    order rank by first appearance in the window (equalities preserved,
                                        numeric distance between raw IDs discarded; missing IDs get their own rank)
    {split}_ids.npy  int64   (n,)       obs_id, in file order
    train_y.npy      int16   (n,)       class index into meta['classes']
    meta.json                          counts, vocabularies, unknown counts, tick report, source fingerprint
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

NUM = ['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux']
CATS = ['venue', 'action', 'side', 'trade']
RAW = ['obs_id', 'order_id'] + CATS + NUM
FIXED = {'action': {'A': 1, 'D': 2, 'U': 3}, 'side': {'A': 1, 'B': 2},
         'trade': {'FALSE': 1, '0': 1, 'TRUE': 2, '1': 2}}
L = 100


def norm_strings(series):
    s = series.astype('string').str.strip().str.upper()
    return s.str.replace(r'\.0$', '', regex=True).fillna('<MISSING>')


def find_files(root='/kaggle/input', **explicit):
    tokens = {'train': 'x_train', 'target': 'y_train', 'test': 'x_test'}
    need = {'train': set(RAW), 'test': set(RAW), 'target': {'obs_id', 'eqt_code_cat'}}
    out = {}
    for key, token in tokens.items():
        if explicit.get(key):
            out[key] = Path(explicit[key]); continue
        hits = [p for p in Path(root).rglob('*.csv') if token in p.name.lower()
                and need[key].issubset(pd.read_csv(p, nrows=0).columns)]
        if len(hits) != 1:
            raise ValueError(f'{key}: {len(hits)} candidate CSV files, pass the path explicitly: {hits}')
        out[key] = hits[0]
    return out


def estimate_tick(path, rows=2_000_000):
    """Most frequent small positive spread, then check that prices sit on that grid."""
    f = pd.read_csv(path, usecols=['price', 'bid', 'ask'], nrows=rows)
    s = np.round((f.ask - f.bid).to_numpy(float), 6)
    s = s[np.isfinite(s) & (s > 0)]
    counts = pd.Series(s).value_counts()
    frequent = counts[counts >= 0.01 * len(s)]
    tick = float(frequent.index.min() if len(frequent) else counts.index[0])
    report = {'tick': tick, 'top_spreads': {str(k): int(v) for k, v in counts.head(8).items()}}
    for c in ['price', 'bid', 'ask']:
        u = f[c].to_numpy(float) / tick
        report[f'{c}_off_grid'] = float(np.nanmean(np.abs(u - np.round(u)) > 1e-3))
    report['grid_ok'] = max(report[f'{c}_off_grid'] for c in ['bid', 'ask']) < 0.01
    return report


def local_order_rank(oid):
    """(n,100) raw order ids → rank of first appearance per window. Vectorised.

    Missing ids never match each other. Raises on non-integer ids."""
    n = len(oid)
    oid = np.asarray(oid, dtype=float)
    miss = ~np.isfinite(oid)
    if (oid[~miss] != np.floor(oid[~miss])).any():
        raise ValueError('order_id must be integer-valued')
    pos = np.broadcast_to(np.arange(L), oid.shape)
    o = np.where(miss, -1 - pos, oid).astype(np.int64)
    if np.abs(o).max(initial=0) >= 2 ** 30:
        raise ValueError('order_id too large for the vectorised remap')
    key = (np.arange(n)[:, None].astype(np.int64) << 32) + (o + 2 ** 31)
    _, first, inv = np.unique(key.ravel(), return_index=True, return_inverse=True)
    urow, upos = first // L, first % L
    order = np.lexsort((upos, urow))
    rows_sorted = urow[order]
    starts = np.r_[0, np.flatnonzero(np.diff(rows_sorted)) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order)) - np.repeat(starts, lengths)
    return rank[inv].reshape(n, L).astype(np.int8)


def encode_cats(frame, venue_vocab, unknown):
    n = len(frame) // L
    out = np.zeros((n, L, 4), dtype=np.int8)
    for j, c in enumerate(CATS):
        mapping = {v: i + 1 for i, v in enumerate(venue_vocab)} if c == 'venue' else FIXED[c]
        s = norm_strings(frame[c])
        codes = s.map(mapping)
        unknown[c] = unknown.get(c, 0) + int(codes.isna().sum())
        out[:, :, j] = codes.fillna(0).to_numpy(int).reshape(n, L)
    return out


def build(files, lab_dir, tick='auto', chunk_windows=20000):
    lab_dir = Path(lab_dir); raw = lab_dir / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    files = {k: Path(v) for k, v in files.items()}
    source = {k: [str(v.resolve()), v.stat().st_size] for k, v in files.items()}
    meta_path = raw / 'meta.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['source'] != source:
            raise ValueError('Raw cache built from other files: choose another lab_dir')
        return meta
    # pass 1: counts and venue vocabulary (train only; test venues outside it become unknown=0)
    counts, venues = {}, set()
    for split in ['train', 'test']:
        counts[split] = 0
        for f in pd.read_csv(files[split], usecols=['obs_id', 'venue'], chunksize=chunk_windows * L):
            counts[split] += len(f)
            if split == 'train':
                venues.update(norm_strings(f.venue).unique().tolist())
        if counts[split] % L:
            raise ValueError(f'{split}: row count not divisible by {L}')
        counts[split] //= L
    venues.discard('<MISSING>')
    venue_vocab = sorted(venues, key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v))
    tick_report = estimate_tick(files['train']) if tick == 'auto' else {'tick': float(tick), 'given': True}
    unknown = {}
    for split in ['train', 'test']:
        n = counts[split]
        arr = {'num': np.lib.format.open_memmap(raw / f'{split}_num.npy', 'w+', np.float32, (n, L, 6)),
               'cat': np.lib.format.open_memmap(raw / f'{split}_cat.npy', 'w+', np.int8, (n, L, 4)),
               'oid': np.lib.format.open_memmap(raw / f'{split}_oid.npy', 'w+', np.int8, (n, L)),
               'ids': np.lib.format.open_memmap(raw / f'{split}_ids.npy', 'w+', np.int64, (n,))}
        cur, unk = 0, {}
        for f in pd.read_csv(files[split], usecols=RAW, chunksize=chunk_windows * L):
            m = len(f) // L
            ids = pd.to_numeric(f.obs_id, errors='raise').to_numpy().reshape(m, L)
            if not (ids == ids[:, :1]).all():
                raise ValueError(f'{split}: windows are not 100 contiguous rows of one obs_id (rows {cur * L}+)')
            if (ids[:, 0] != np.floor(ids[:, 0])).any():
                raise ValueError('obs_id must be integer')
            arr['ids'][cur:cur + m] = ids[:, 0]
            arr['num'][cur:cur + m] = f[NUM].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32).reshape(m, L, 6)
            arr['cat'][cur:cur + m] = encode_cats(f, venue_vocab, unk)
            arr['oid'][cur:cur + m] = local_order_rank(f.order_id.to_numpy().reshape(m, L))
            cur += m
            print(f'[raw {split}] {cur:,}/{n:,}', flush=True)
        if len(np.unique(arr['ids'])) != n:
            raise ValueError(f'{split}: repeated obs_id across windows')
        for a in arr.values():
            a.flush()
        unknown[split] = unk
        del arr
    target = pd.read_csv(files['target'])
    if target.obs_id.duplicated().any():
        raise ValueError('duplicate obs_id in y_train')
    ids = np.load(raw / 'train_ids.npy')
    if set(ids) != set(target.obs_id):
        raise ValueError('X_train and y_train obs_id sets differ')
    labels = target.set_index('obs_id').loc[ids, 'eqt_code_cat'].to_numpy()
    classes, y = np.unique(labels, return_inverse=True)
    np.save(raw / 'train_y.npy', y.astype(np.int16))
    meta = {'source': source, 'counts': counts, 'venue_vocab': venue_vocab, 'unknown_codes': unknown,
            'classes': classes.tolist(), 'tick_report': tick_report, 'tick': tick_report['tick']}
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return meta


def load(lab_dir, split):
    raw = Path(lab_dir) / 'raw'
    out = {k: np.load(raw / f'{split}_{k}.npy', mmap_mode='r') for k in ['num', 'cat', 'oid', 'ids']}
    if split == 'train':
        out['y'] = np.load(raw / 'train_y.npy')
    return out


def meta(lab_dir):
    return json.loads((Path(lab_dir) / 'raw' / 'meta.json').read_text())
