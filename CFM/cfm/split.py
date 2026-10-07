"""Partitions, stratified by class. Drawn once, saved, reused by every run.

Order of draws per class:
1. audit  — random, drawn FIRST from a dedicated seed: identical whatever the stress settings. Closed.
2. stress — tail of median log depth (in lots). Direction 'auto' follows the test median (transductive:
            uses unlabeled test covariates; recorded in split.json), or forced 'low' / 'high'.
3. valid  — `valid_mode='random'`: random among the rest (rounds 1–2; optimistic, shares stock-days with fit).
            `valid_mode='cluster'` (V3): per class, k-means on window REGIME features (depth, lots, ticks,
            venue mix) and whole clusters are held out until the valid share is reached. Windows of one
            stock-day share a regime, so they tend to fall in the same cluster: valid then contains mostly
            regimes the fit has not seen — an approximation of unseen days, not a chronological split.
4. fit    — everything else.
Neither stress nor valid is chronological: no dates are available.
"""
import json
from pathlib import Path
import numpy as np
from . import io
from .registry import compute


def make(lab_dir, cfg):
    s = cfg['split']
    path = Path(lab_dir) / 'split.npz'
    if path.exists():
        saved = json.loads((Path(lab_dir) / 'split.json').read_text())
        if saved['config'] != s:
            raise ValueError('Split already exists with another config: keep it, or use a new lab_dir')
        return load(lab_dir)
    raw = io.load(lab_dir, 'train')
    y = np.asarray(raw['y'])
    lv_train, names = compute('levels', lab_dir, 'train', cfg['lot'])
    lv_test, _ = compute('levels', lab_dir, 'test', cfg['lot'])
    j = names.index('log_depth_q50')
    depth, depth_test = np.asarray(lv_train[:, j]), np.asarray(lv_test[:, j])
    tail = s['stress_tail']
    if tail == 'auto':
        tail = 'low' if np.nanmedian(depth_test) < np.nanmedian(depth) else 'high'
    mode = s.get('valid_mode', 'random')
    if mode not in ('random', 'cluster'):
        raise ValueError('valid_mode must be random or cluster')
    regime = _regime_matrix(lab_dir, cfg, s) if mode == 'cluster' else None
    rng_audit = np.random.default_rng(s['seed'] + 10_000)
    rng = np.random.default_rng(s['seed'])
    parts = {k: [] for k in ['fit', 'valid', 'stress', 'audit']}
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n = len(idx)
        na, ns, nv = int(s['audit'] * n), int(s['stress'] * n), int(s['valid'] * n)
        audit = rng_audit.permutation(idx)[:na]
        rest = np.setdiff1d(idx, audit)
        d = np.nan_to_num(depth[rest], nan=np.nanmedian(depth))
        ordered = rest[np.argsort(d, kind='stable')]
        stress = ordered[:ns] if tail == 'low' else ordered[len(ordered) - ns:]
        rest = rng.permutation(np.setdiff1d(rest, stress))
        if mode == 'cluster':
            valid = _cluster_holdout(regime[rest], rest, nv, s.get('clusters_per_class', 40), rng)
        else:
            valid = rest[:nv]
        parts['audit'] += audit.tolist(); parts['stress'] += stress.tolist()
        parts['valid'] += valid.tolist(); parts['fit'] += np.setdiff1d(rest, valid).tolist()
    parts = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in parts.items()}
    allidx = np.concatenate(list(parts.values()))
    assert len(allidx) == len(y) == len(np.unique(allidx)), 'partitions must be disjoint and complete'
    np.savez(path, **parts)
    info = {'config': s, 'tail_used': tail, 'sizes': {k: int(len(v)) for k, v in parts.items()},
            'transductive': s['stress_tail'] == 'auto', 'valid_mode': mode,
            'median_log_depth': {'train': float(np.nanmedian(depth)), 'test': float(np.nanmedian(depth_test))}}
    (Path(lab_dir) / 'split.json').write_text(json.dumps(info, indent=2))
    print('split', info, flush=True)
    return parts


def _regime_matrix(lab_dir, cfg, s):
    from .registry import window_matrix
    X, _, _ = window_matrix(s.get('regime_blocks', ['levels', 'lots', 'ticks', 'cat_freq']), lab_dir, 'train', cfg['lot'])
    X = np.asarray(X, dtype=np.float64)
    m, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    return np.nan_to_num((X - m) / np.where(sd > 1e-9, sd, 1), nan=0.).astype(np.float32)


def _cluster_holdout(Z, rows, n_target, k, rng):
    """k-means on one class's regime features, then hold out random whole clusters until n_target rows."""
    from sklearn.cluster import KMeans
    k = max(2, min(k, len(rows) // 5))
    lab = KMeans(n_clusters=k, n_init=3, random_state=int(rng.integers(1 << 31))).fit_predict(Z)
    chosen, total = [], 0
    for c in rng.permutation(k):
        if total >= n_target:
            break
        members = rows[lab == c]
        chosen.append(members); total += len(members)
    return np.sort(np.concatenate(chosen)) if chosen else rows[:0]


def load(lab_dir):
    z = np.load(Path(lab_dir) / 'split.npz')
    return {k: z[k] for k in z.files}


def subsample(idx, y, frac, seed):
    """Stratified subsample of indices (for fast screening)."""
    if frac >= 1:
        return idx
    rng = np.random.default_rng(seed)
    out = [rng.permutation(idx[y[idx] == c])[:max(1, int(frac * (y[idx] == c).sum()))] for c in np.unique(y[idx])]
    return np.sort(np.concatenate(out))
