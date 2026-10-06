"""Neighbour smoothing: let windows that look alike vote together. TRANSDUCTIVE — uses unlabeled X_test.

Idea: the test holds ~20 windows per stock and day. Windows of the same stock-day share a regime
(depth, flux size, odd-lot share, spread, venue mix). If a representation puts them close together,
averaging each window's probabilities with its neighbours' corrects isolated mistakes.

Protocol:
1. neighbour purity on fit (labels): share of k nearest neighbours with the same class, per representation.
   Tells which representation groups same-stock windows, before any smoothing.
2. smoothing grid on valid ∪ stress (dev models only, neighbours searched INSIDE that pool):
   a SELECTION score — the chosen setting is then applied once to the test.
3. Pool vs test density: valid ∪ stress keeps ~25 % of each stock-day (≈5 windows), the test keeps all 20.
   k is therefore scaled by `k_scale` (default 4) when moving to the test. This is an assumption, written down.
"""
import numpy as np
import pandas as pd
import torch
from .blend import sinkhorn_balance
from .registry import window_matrix

GROUPING_BLOCKS = ['levels', 'ticks', 'lots', 'cat_freq', 'position']


def knn(Z, k, device=None, chunk=4096):
    """Indices of the k nearest neighbours (cosine), self excluded. Exact, chunked, GPU if available."""
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    k = min(k, len(Z) - 1)
    Z = torch.as_tensor(np.asarray(Z, dtype=np.float32), device=dev)
    Z = torch.nn.functional.normalize(Z, dim=1)
    out = []
    for i in range(0, len(Z), chunk):
        s = Z[i:i + chunk] @ Z.T
        s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -2       # exclude self
        out.append(s.topk(k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def smooth(P, nbrs, alpha, iters=1):
    """Label propagation: Q ← (1-α)·P + α·mean(Q[neighbours]), `iters` times."""
    Q = P.copy()
    for _ in range(iters):
        Q = (1 - alpha) * P + alpha * Q[nbrs].mean(1)
    return Q / Q.sum(1, keepdims=True)


def standardise(X):
    """Within-set standardisation for neighbour search only (NaN → 0)."""
    m, s = np.nanmean(X, 0), np.nanstd(X, 0)
    return np.nan_to_num((X - m) / np.where(s > 1e-9, s, 1), nan=0.)


def window_rep(lab_dir, cfg, split, rows, blocks=GROUPING_BLOCKS):
    X, _, _ = window_matrix(blocks, lab_dir, split, cfg['lot'])
    return standardise(np.asarray(X)[rows])


def emb_rep(lab_dir, runs, weights, part):
    """Mean of L2-normalised embeddings across runs (same windows, checked)."""
    zs, ids = [], None
    for r in runs:
        z = np.load(f'{lab_dir}/runs/{r}/emb_{weights}_{part}.npz')
        if ids is not None and not np.array_equal(ids, z['obs_ids']):
            raise ValueError(f'{r}: embeddings rows differ')
        ids = z['obs_ids']
        v = z['z'].astype(np.float32)
        zs.append(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9))
    return np.mean(zs, 0), ids


def purity(Z, y, ks=(5, 10, 20)):
    nb = knn(Z, max(ks))
    return {f'purity@{k}': float((y[nb[:, :k]] == y[:, None]).mean()) for k in ks}


def grid(P, y, reps, ks=(3, 5, 10, 20), alphas=(.3, .5, .8), iters=(1, 5)):
    """Accuracy after neighbour smoothing (+ balance) for each representation. Higher is better."""
    rows = [{'rep': '(none)', 'k': 0, 'alpha': 0, 'iters': 0, 'acc': float((P.argmax(1) == y).mean()),
             'acc_balanced': float((sinkhorn_balance(P).argmax(1) == y).mean())}]
    for name, Z in reps.items():
        nb_all = knn(Z, max(ks))
        for k in [k for k in ks if k < len(Z)]:
            for a in alphas:
                for it in iters:
                    Q = smooth(P, nb_all[:, :k], a, it)
                    rows.append({'rep': name, 'k': k, 'alpha': a, 'iters': it,
                                 'acc': float((Q.argmax(1) == y).mean()),
                                 'acc_balanced': float((sinkhorn_balance(Q).argmax(1) == y).mean())})
    return pd.DataFrame(rows).sort_values('acc_balanced', ascending=False)


def apply(P, Z, k, alpha, iters, k_scale=4, balance=True):
    """Apply a setting chosen on valid ∪ stress to the test (k scaled to the test density)."""
    kt = min(max(1, int(round(k * k_scale))), len(Z) - 1)
    Q = smooth(P, knn(Z, kt), alpha, iters)
    return (sinkhorn_balance(Q) if balance else Q), kt
