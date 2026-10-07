"""Blend saved probabilities, balance them (optional, transductive), write and check the submission."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from . import io
from .metrics import ledger, scores


def load_probs(lab_dir, runs, part):
    """Load <run>/<part>.npz for each run and check that rows are the same windows in the same order."""
    out, ref = [], None
    for r in runs:
        z = np.load(Path(lab_dir) / 'runs' / r / f'{part}.npz')
        if ref is None:
            ref = z['obs_ids']
        elif not np.array_equal(ref, z['obs_ids']):
            raise ValueError(f'{r}/{part}: obs_ids differ from {runs[0]}')
        out.append(z['p'])
    y = z['y'] if 'y' in z.files else None
    return np.stack(out), y, ref


def weights_on_valid(P, y):
    """Softmax-parametrised weights minimising valid log-loss. Valid becomes a SELECTION score."""
    def loss(w):
        w = np.exp(w - w.max()); w /= w.sum()
        return scores(y, np.tensordot(w, P, 1))['logloss']
    w = minimize(loss, np.zeros(len(P)), method='Nelder-Mead', options={'maxiter': 400}).x
    w = np.exp(w - w.max())
    return w / w.sum()


def sinkhorn_balance(p, iters=50, target=None):
    """Rescale class columns so that the predicted class mass matches `target` (default uniform).
    Uses the unlabeled set as a whole: only valid if the rules allow it and the set is ~balanced."""
    k = p.shape[1]
    target = np.full(k, len(p) / k) if target is None else target
    q = np.clip(p, 1e-12, None).copy()
    for _ in range(iters):
        q *= (target / q.sum(0))[None]
        q /= q.sum(1, keepdims=True)
    return q


def report(lab_dir, runs, optimise=False):
    P, y, _ = load_probs(lab_dir, runs, 'valid')
    S, ys, _ = load_probs(lab_dir, runs, 'stress')
    rows = [{'model': r, **{f'valid_{k}': v for k, v in scores(y, P[i]).items()},
             **{f'stress_{k}': v for k, v in scores(ys, S[i]).items()}} for i, r in enumerate(runs)]
    w = weights_on_valid(P, y) if optimise else np.full(len(runs), 1 / len(runs))
    pb, sb = np.tensordot(w, P, 1), np.tensordot(w, S, 1)
    rows.append({'model': 'blend' + (' (weights fitted on valid)' if optimise else ' (mean)'),
                 **{f'valid_{k}': v for k, v in scores(y, pb).items()},
                 **{f'stress_{k}': v for k, v in scores(ys, sb).items()}})
    rows.append({'model': 'blend + balance', **{f'valid_{k}': v for k, v in scores(y, sinkhorn_balance(pb)).items()},
                 **{f'stress_{k}': v for k, v in scores(ys, sinkhorn_balance(sb)).items()}})
    pred = P.argmax(2)
    agree = pd.DataFrame([[float((pred[i] == pred[j]).mean()) for j in range(len(runs))] for i in range(len(runs))],
                         index=runs, columns=runs)
    oracle = float((pred == y[None]).any(0).mean())
    return pd.DataFrame(rows), w, agree, oracle


def submission(lab_dir, runs, weights=None, source='test_refit', balance=False, name='submission'):
    lab_dir = Path(lab_dir)
    meta = io.meta(lab_dir)
    P, _, ids = load_probs(lab_dir, runs, source)
    w = np.full(len(runs), 1 / len(runs)) if weights is None else np.asarray(weights)
    p = np.tensordot(w, P, 1)
    if balance:
        p = sinkhorn_balance(p)
    test_ids = np.asarray(io.load(lab_dir, 'test')['ids'])
    if not np.array_equal(ids, test_ids):
        raise ValueError('prediction rows are not aligned with test obs_ids')
    if not np.allclose(p.sum(1), 1, atol=1e-4) or p.shape[1] != len(meta['classes']):
        raise ValueError('probabilities must be normalised over all classes')
    sub = pd.DataFrame({'obs_id': ids, 'eqt_code_cat': np.asarray(meta['classes'])[p.argmax(1)]})
    path = lab_dir / f'{name}.csv'
    sub.to_csv(path, index=False)
    np.savez(lab_dir / f'{name}_probs.npz', obs_ids=ids, p=p, classes=np.asarray(meta['classes']))
    info = {'runs': runs, 'weights': list(map(float, w)), 'source': source, 'balance': balance, 'file': str(path),
            'class_counts': sub.eqt_code_cat.value_counts().sort_index().to_dict()}
    (lab_dir / f'{name}.json').write_text(json.dumps(info, indent=2, default=str))
    ledger(lab_dir, {'name': name, 'kind': 'submission', **{k: v for k, v in info.items() if k != 'class_counts'}})
    return path
