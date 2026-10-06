"""Label-free indicators on the test, to rank models without spending leaderboard submissions.

None of them is an accuracy. They are only useful once anchored to a few leaderboard scores
(see LEADERBOARD.md): a proxy that orders the submitted models like the leaderboard does can then
be trusted a little to pre-select the next ones.

- confidence      : mean max probability (falls under shift; also depends on calibration)
- entropy         : mean predictive entropy
- balance_kl      : KL(predicted class shares || uniform). The test looks balanced: 0 is ideal
- max_min_ratio   : most / least predicted class count
- seed_agreement  : share of identical argmax with the other seed(s) of the same config
"""
from pathlib import Path
import numpy as np
import pandas as pd


def _p(lab_dir, run, part):
    z = np.load(Path(lab_dir) / 'runs' / run / f'{part}.npz')
    return z['p'], (z['y'] if 'y' in z.files else None)


def indicators(p):
    k = p.shape[1]
    share = np.bincount(p.argmax(1), minlength=k) / len(p)
    ent = -(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return {'confidence': float(p.max(1).mean()), 'entropy': float(ent.mean()),
            'balance_kl': float((share * np.log(np.clip(share * k, 1e-12, None))).sum()),
            'max_min_ratio': float(share.max() / max(share.min(), 1e-12))}


def table(lab_dir, runs, part='test_dev'):
    """One row per run: indicators on `part` (default test_dev = dev models on the test) and on stress."""
    rows, preds = [], {}
    for r in runs:
        p, _ = _p(lab_dir, r, part)
        ps, ys = _p(lab_dir, r, 'stress')
        preds[r] = p.argmax(1)
        rows.append({'run': r, **{f'test_{k}': v for k, v in indicators(p).items()},
                     'stress_acc': float((ps.argmax(1) == ys).mean()), 'stress_confidence': float(ps.max(1).mean())})
    df = pd.DataFrame(rows).set_index('run')
    fam = {r: r.rsplit('_s', 1)[0] for r in runs}
    df['test_seed_agreement'] = [np.mean([(preds[r] == preds[o]).mean() for o in runs if o != r and fam[o] == fam[r]])
                                 if sum(fam[o] == fam[r] for o in runs) > 1 else np.nan for r in runs]
    return df
