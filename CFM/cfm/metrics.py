"""Metrics, paired bootstrap and the run ledger."""
import datetime as dt
import json
from pathlib import Path
import numpy as np
import pandas as pd


def scores(y, p):
    p = np.clip(p, 1e-12, 1)
    p = p / p.sum(1, keepdims=True)
    return {'acc': float((p.argmax(1) == y).mean()),
            'logloss': float(-np.log(p[np.arange(len(y)), y]).mean())}


def per_class_recall(y, p, k=None):
    k = k or p.shape[1]
    pred = p.argmax(1)
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(k)])


def paired_bootstrap(y, p_base, p_new, n=1000, seed=0):
    """CI of accuracy(new) - accuracy(base) on the SAME windows. Windows are resampled as if independent:
    windows from the same stock-day are not, so this interval is optimistic (too narrow)."""
    d = (p_new.argmax(1) == y).astype(float) - (p_base.argmax(1) == y).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def ledger(lab_dir, row):
    """Append one line per run/screen to <lab>/ledger.csv: the history of every decision."""
    path = Path(lab_dir) / 'ledger.csv'
    row = {'time': dt.datetime.now().isoformat(timespec='seconds'), **{
        k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}}
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
