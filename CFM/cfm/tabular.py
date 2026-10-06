"""Tabular models on window blocks: xgb (GPU if asked), lgbm, mlp (torch). Same interface for all.

Early stopping uses an INNER slice of the fitting split (10%), never valid: valid stays a clean score
for screening comparisons, stress stays a diagnostic.
"""
import json
import time
from pathlib import Path
import numpy as np
from . import io, split as splitlib
from .config import digest
from .metrics import ledger, scores
from .registry import window_matrix

DEFAULTS = {
    'xgb': {'n_estimators': 2000, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8,
            'colsample_bytree': 0.6, 'min_child_weight': 2, 'max_bin': 128, 'early_stopping_rounds': 40},
    'lgbm': {'n_estimators': 2000, 'learning_rate': 0.08, 'num_leaves': 63, 'subsample': 0.8,
             'subsample_freq': 1, 'colsample_bytree': 0.6, 'min_child_samples': 20, 'early_stopping_rounds': 40},
    'mlp': {'hidden': 512, 'dropout': 0.2, 'lr': 2e-3, 'weight_decay': 1e-4, 'epochs': 60,
            'batch_size': 1024, 'patience': 6},
}


def inner_split(fit_idx, y, seed, frac=0.1):
    es = splitlib.subsample(fit_idx, y, frac, seed + 77)
    return np.setdiff1d(fit_idx, es), es


class Standardizer:
    def __init__(self, X):
        self.mean = np.nanmean(X, 0)
        self.std = np.nanstd(X, 0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-8), self.std, 1)

    def __call__(self, X):
        return np.clip(np.nan_to_num((X - self.mean) / self.std, nan=0.), -8, 8).astype(np.float32)


def _mlp(Xtr, ytr, Xes, yes, k, params, seed, device):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dev = torch.device(device)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], params['hidden']), nn.GELU(), nn.Dropout(params['dropout']),
                        nn.Linear(params['hidden'], params['hidden']), nn.GELU(), nn.Dropout(params['dropout']),
                        nn.Linear(params['hidden'], k)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev, dtype=torch.long)
    Xe = torch.tensor(Xes, device=dev)
    best, best_loss, stale = None, np.inf, 0
    g = torch.Generator(device='cpu').manual_seed(seed)
    for epoch in range(params['epochs']):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(params['batch_size']):
            b = b.to(dev)
            opt.zero_grad()
            nn.functional.cross_entropy(net(Xt[b]), yt[b], label_smoothing=0.02).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pe = net(Xe).softmax(-1).cpu().numpy()
        loss = scores(yes, pe)['logloss']
        if loss < best_loss:
            best_loss, stale, best = loss, 0, ({k_: v.clone() for k_, v in net.state_dict().items()}, epoch + 1)
        else:
            stale += 1
            if stale >= params['patience']:
                break
    net.load_state_dict(best[0]); net.eval()

    def predict(X):
        with torch.no_grad():
            return np.concatenate([net(torch.tensor(X[i:i + 8192], device=dev)).softmax(-1).cpu().numpy()
                                   for i in range(0, len(X), 8192)])
    return predict, best[1]


def train(X, y, tr, es, model, params, seed, device, k=None):
    """Return (predict_fn, best_iteration). X may contain NaN."""
    k = k or int(y.max()) + 1
    p = {**DEFAULTS[model], **(params or {})}
    if model == 'xgb':
        import xgboost as xgb
        rounds = p.pop('early_stopping_rounds')
        clf = xgb.XGBClassifier(objective='multi:softprob', tree_method='hist', device=device,
                                random_state=seed, eval_metric='mlogloss', early_stopping_rounds=rounds, **p)
        clf.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], verbose=False)
        return clf.predict_proba, int(clf.best_iteration) + 1
    if model == 'lgbm':
        import lightgbm as lgb
        rounds = p.pop('early_stopping_rounds')
        clf = lgb.LGBMClassifier(objective='multiclass', random_state=seed, verbose=-1, **p)
        clf.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], callbacks=[lgb.early_stopping(rounds, verbose=False)])
        return clf.predict_proba, int(clf.best_iteration_ or p['n_estimators'])
    if model == 'mlp':
        st = Standardizer(X[tr])
        fn, it = _mlp(st(X[tr]), y[tr], st(X[es]), y[es], k, p, seed, device)
        return (lambda Z: fn(st(Z))), it
    raise ValueError(model)


def refit_train(X, y, idx, model, params, seed, device, best_iter):
    """All labelled rows, fixed iteration count = best development iteration (no early stopping)."""
    p = {**DEFAULTS[model], **(params or {})}
    p.pop('early_stopping_rounds', None)
    if model == 'xgb':
        import xgboost as xgb
        p['n_estimators'] = best_iter
        clf = xgb.XGBClassifier(objective='multi:softprob', tree_method='hist', device=device, random_state=seed, **p)
        clf.fit(X[idx], y[idx]); return clf.predict_proba
    if model == 'lgbm':
        import lightgbm as lgb
        p['n_estimators'] = best_iter
        clf = lgb.LGBMClassifier(objective='multiclass', random_state=seed, verbose=-1, **p)
        clf.fit(X[idx], y[idx]); return clf.predict_proba
    if model == 'mlp':
        p = {**p, 'epochs': best_iter, 'patience': 10 ** 9}
        st = Standardizer(X[idx])
        fn, _ = _mlp(st(X[idx]), y[idx], st(X[idx[:2000]]), y[idx[:2000]], int(y.max()) + 1, p, seed, device)
        return lambda Z: fn(st(Z))
    raise ValueError(model)


def run(lab_dir, cfg, blocks, model='xgb', params=None, seed=0, name=None, refit=False, note=''):
    """Train on fit, score valid + stress, save probabilities. Optional refit on all labels → test."""
    lab_dir = Path(lab_dir)
    name = name or f'tab_{model}_{digest([blocks, params, seed], 8)}'
    out = lab_dir / 'runs' / name
    out.mkdir(parents=True, exist_ok=True)
    raw = io.load(lab_dir, 'train'); y = np.asarray(raw['y']).astype(int)
    sp = splitlib.load(lab_dir)
    X, cols, _ = window_matrix(blocks, lab_dir, 'train', cfg['lot'])
    tr, es = inner_split(sp['fit'], y, seed)
    t0 = time.time()
    device = cfg['device'] if model in ('xgb', 'mlp') else 'cpu'
    predict, best_iter = train(X, y, tr, es, model, params, seed, device)
    result = {'name': name, 'kind': 'tabular', 'model': model, 'blocks': blocks, 'params': params or {},
              'seed': seed, 'n_features': X.shape[1], 'best_iter': best_iter, 'seconds': round(time.time() - t0, 1)}
    for part in ['valid', 'stress']:
        p = predict(X[sp[part]])
        np.savez(out / f'{part}.npz', idx=sp[part], obs_ids=np.asarray(raw['ids'])[sp[part]], y=y[sp[part]], p=p)
        result.update({f'{part}_{k}': v for k, v in scores(y[sp[part]], p).items()})
    np.savez(out / 'sealed_audit.npz', idx=sp['audit'], obs_ids=np.asarray(raw['ids'])[sp['audit']],
             p=predict(X[sp['audit']]))  # scored only by cfm.audit.open_audit
    Xt, _, _ = window_matrix(blocks, lab_dir, 'test', cfg['lot'])
    test_ids = np.asarray(io.load(lab_dir, 'test')['ids'])
    np.savez(out / 'test_dev.npz', obs_ids=test_ids, p=predict(Xt))
    if refit:
        labelled = np.arange(len(y))
        fn = refit_train(X, y, labelled, model, params, seed, device, best_iter)
        np.savez(out / 'test_refit.npz', obs_ids=test_ids, p=fn(Xt))
        result['refit'] = 'all labels incl. audit, fixed iterations = best_iter'
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    (out / 'columns.json').write_text(json.dumps(cols))
    ledger(lab_dir, {**{k: v for k, v in result.items() if k != 'params'}, 'params': params or {}, 'note': note})
    print(f"[{name}] valid {result['valid_acc']:.4f} / stress {result['stress_acc']:.4f} "
          f"({X.shape[1]} features, {result['seconds']}s)", flush=True)
    return result
