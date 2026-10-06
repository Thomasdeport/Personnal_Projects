"""The audit split is scored ONCE, after freezing model, preprocessing, seeds, refit budget and blend rule.

Every dev run saves `sealed_audit.npz` (predictions made by the frozen dev model). Nothing reads it
except `open_audit`, which writes an irreversible marker. After that, any further tuning makes the
audit a selection score — say so in your write-up.
"""
import datetime as dt
import json
from pathlib import Path
import numpy as np
from . import io
from .metrics import ledger, scores


def open_audit(lab_dir, runs, weights=None, confirm=False):
    lab_dir = Path(lab_dir)
    marker = lab_dir / 'AUDIT_OPENED.json'
    if not confirm:
        raise PermissionError('Pass confirm=True only once decisions are frozen (see docstring).')
    y = np.asarray(io.load(lab_dir, 'train')['y']).astype(int)
    P, idx = [], None
    for r in runs:
        z = np.load(lab_dir / 'runs' / r / 'sealed_audit.npz')
        if idx is not None and not np.array_equal(idx, z['idx']):
            raise ValueError('audit rows differ between runs')
        idx = z['idx']; P.append(z['p'])
    w = np.full(len(P), 1 / len(P)) if weights is None else np.asarray(weights)
    res = {'runs': runs, 'weights': list(map(float, w)), **scores(y[idx], np.tensordot(w, np.stack(P), 1)),
           'individual': {r: scores(y[idx], p) for r, p in zip(runs, P)},
           'time': dt.datetime.now().isoformat(timespec='seconds')}
    history = json.loads(marker.read_text()) if marker.exists() else []
    if history:
        print(f'WARNING: audit already opened {len(history)} time(s): it is no longer an independent score.')
    marker.write_text(json.dumps(history + [res], indent=2))
    ledger(lab_dir, {'name': 'AUDIT', 'kind': 'audit', 'runs': runs, 'acc': res['acc'], 'logloss': res['logloss'],
                     'opened_before': len(history)})
    return res
