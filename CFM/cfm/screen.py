"""Feature screening funnel: cheap → expensive. Each stage filters what the next one has to train.

1. univariate : per column, η² of the class on the FIT split (utility), KS train/test (drift),
                max |corr| with the base set (redundancy), degenerate share. Seconds.
2. adversarial: per block, AUC of a train-vs-test classifier on that block alone. Where does the shift live?
3. forward    : base + one candidate block, retrained; Δ valid acc with paired bootstrap, Δ stress. Minutes.
4. ablate     : full set minus one block, retrained. Confirms that kept blocks still pay their way together.

All stages use fit/valid/stress only. Audit is never read here.
A masked-at-inference importance is a SENSITIVITY test, not an ablation: stages 3–4 always retrain.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from . import io, split as splitlib
from .metrics import ledger, paired_bootstrap, scores
from .registry import BLOCKS, window_matrix
from .tabular import inner_split, train


def eta2(X, y):
    """Share of variance explained by the class, per column (NaN → column mean)."""
    X = np.where(np.isfinite(X), X, np.nanmean(X, 0))
    grand = X.mean(0)
    total = ((X - grand) ** 2).sum(0)
    between = sum((y == c).sum() * (X[y == c].mean(0) - grand) ** 2 for c in np.unique(y))
    return np.where(total > 1e-12, between / np.maximum(total, 1e-12), 0)


def univariate(lab_dir, cfg, blocks, base=('base_relative', 'cat_freq'), max_rows=60000, seed=0):
    y = np.asarray(io.load(lab_dir, 'train')['y']).astype(int)
    sp = splitlib.load(lab_dir)
    fit = splitlib.subsample(sp['fit'], y, max_rows / len(sp['fit']), seed)
    Xb, _, _ = window_matrix(list(base), lab_dir, 'train', cfg['lot'])
    Xb = np.nan_to_num(Xb[fit])
    Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-9)
    rng = np.random.default_rng(seed)
    rows = []
    for b in blocks:
        X, names, _ = window_matrix([b], lab_dir, 'train', cfg['lot'])
        Xt, _, _ = window_matrix([b], lab_dir, 'test', cfg['lot'])
        e = eta2(X[fit], y[fit])
        ti = rng.choice(len(Xt), min(len(Xt), max_rows), replace=False)
        for j, n in enumerate(names):
            a, t = X[fit, j], Xt[ti, j]
            a, t = a[np.isfinite(a)], t[np.isfinite(t)]
            col = np.nan_to_num(X[fit, j])
            z = (col - col.mean()) / (col.std() + 1e-9)
            red = float(np.abs(z @ Xb / len(z)).max()) if b not in base else np.nan
            rows.append({'block': b, 'family': BLOCKS[b].family, 'feature': n, 'eta2': float(e[j]),
                         'ks': float(ks_2samp(a, t).statistic) if len(a) and len(t) else np.nan,
                         'shift_sd': float((t.mean() - a.mean()) / (a.std() + 1e-9)) if len(a) and len(t) else np.nan,
                         'max_corr_base': red, 'nan_share': float(np.mean(~np.isfinite(X[fit, j]))),
                         'mode_share': float(pd.Series(np.round(a, 6)).value_counts(normalize=True).iloc[0]) if len(a) else 1.})
    df = pd.DataFrame(rows).sort_values('eta2', ascending=False)
    df.to_csv(f'{lab_dir}/screen_univariate.csv', index=False)
    return df


def adversarial(lab_dir, cfg, blocks, n=30000, seed=0, device=None):
    """AUC train-vs-test of each block alone (3-fold). 0.5 = indiscernible. High AUC ≠ 'drop it'."""
    import xgboost as xgb
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    device = device or cfg['device']
    rng = np.random.default_rng(seed)
    rows = []
    sets = [[b] for b in blocks] + [list(blocks)]
    for bs in sets:
        X, _, _ = window_matrix(bs, lab_dir, 'train', cfg['lot'])
        Xt, _, _ = window_matrix(bs, lab_dir, 'test', cfg['lot'])
        a = X[rng.choice(len(X), min(n, len(X)), replace=False)]
        t = Xt[rng.choice(len(Xt), min(n, len(Xt)), replace=False)]
        Z, d = np.vstack([a, t]), np.r_[np.zeros(len(a)), np.ones(len(t))]
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, tree_method='hist',
                                device=device, random_state=seed)
        p = cross_val_predict(clf, Z, d, cv=3, method='predict_proba')[:, 1]
        rows.append({'blocks': '+'.join(bs) if len(bs) > 1 else bs[0], 'n_features': Z.shape[1],
                     'auc_train_vs_test': float(roc_auc_score(d, p))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{lab_dir}/screen_adversarial.csv', index=False)
    return df


def _evaluate(lab_dir, cfg, blocks, model, params, seed, fit_frac):
    y = np.asarray(io.load(lab_dir, 'train')['y']).astype(int)
    sp = splitlib.load(lab_dir)
    X, _, _ = window_matrix(list(blocks), lab_dir, 'train', cfg['lot'])
    fit = splitlib.subsample(sp['fit'], y, fit_frac, seed)
    tr, es = inner_split(fit, y, seed)
    device = cfg['device'] if model in ('xgb', 'mlp') else 'cpu'
    predict, it = train(X, y, tr, es, model, params, seed, device)
    out = {'iter': it, 'n_features': X.shape[1]}
    probs = {}
    for part in ['valid', 'stress']:
        probs[part] = predict(X[sp[part]])
        out.update({f'{part}_{k}': v for k, v in scores(y[sp[part]], probs[part]).items()})
    return out, probs, {p: y[sp[p]] for p in ['valid', 'stress']}


def forward(lab_dir, cfg, base, candidates, model='xgb', params=None, seeds=(0,), fit_frac=0.5, tag=''):
    """Base vs base+candidate, retrained, same seeds and same subsample. Paired on the same windows."""
    rows = []
    for seed in seeds:
        ref, pref, ys = _evaluate(lab_dir, cfg, base, model, params, seed, fit_frac)
        rows.append({'candidate': '(base)', 'seed': seed, **ref})
        for c in candidates:
            res, p, _ = _evaluate(lab_dir, cfg, list(base) + [c], model, params, seed, fit_frac)
            d, lo, hi = paired_bootstrap(ys['valid'], pref['valid'], p['valid'], seed=seed)
            ds, los, his = paired_bootstrap(ys['stress'], pref['stress'], p['stress'], seed=seed)
            rows.append({'candidate': c, 'seed': seed, **res, 'd_valid_acc': d, 'd_valid_lo': lo, 'd_valid_hi': hi,
                         'd_valid_logloss': res['valid_logloss'] - ref['valid_logloss'],
                         'd_stress_acc': ds, 'd_stress_lo': los, 'd_stress_hi': his})
            print(f"  +{c:<14} Δvalid {d:+.4f} [{lo:+.4f},{hi:+.4f}]  Δstress {ds:+.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{lab_dir}/screen_forward{tag}.csv', index=False)
    ledger(lab_dir, {'name': f'screen_forward{tag}', 'kind': 'screen', 'model': model, 'blocks': list(base),
                     'candidates': list(candidates), 'seeds': list(seeds), 'fit_frac': fit_frac})
    return df


def ablate(lab_dir, cfg, blocks, model='xgb', params=None, seeds=(0,), fit_frac=0.5, tag=''):
    """Full set vs full set minus one block (retrained). Δ < 0 means the block helps."""
    rows = []
    for seed in seeds:
        ref, pref, ys = _evaluate(lab_dir, cfg, blocks, model, params, seed, fit_frac)
        rows.append({'removed': '(none)', 'seed': seed, **ref})
        for b in blocks:
            rest = [x for x in blocks if x != b]
            res, p, _ = _evaluate(lab_dir, cfg, rest, model, params, seed, fit_frac)
            d, lo, hi = paired_bootstrap(ys['valid'], pref['valid'], p['valid'], seed=seed)
            ds, _, _ = paired_bootstrap(ys['stress'], pref['stress'], p['stress'], seed=seed)
            rows.append({'removed': b, 'seed': seed, **res, 'd_valid_acc': d, 'd_valid_lo': lo,
                         'd_valid_hi': hi, 'd_stress_acc': ds})
            print(f"  -{b:<14} Δvalid {d:+.4f} [{lo:+.4f},{hi:+.4f}]  Δstress {ds:+.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{lab_dir}/screen_ablate{tag}.csv', index=False)
    ledger(lab_dir, {'name': f'screen_ablate{tag}', 'kind': 'screen', 'model': model, 'blocks': list(blocks),
                     'seeds': list(seeds), 'fit_frac': fit_frac})
    return df


def greedy(lab_dir, cfg, base, candidates, model='xgb', params=None, seed=0, fit_frac=0.5, min_gain=0.002):
    """Forward selection: add the best block while Δ valid acc ≥ min_gain AND its bootstrap low > 0.
    Selection on valid → valid becomes a SELECTION score; re-check the final set on stress + another seed."""
    chosen, pool, steps = list(base), list(candidates), []
    while pool:
        df = forward(lab_dir, cfg, chosen, pool, model, params, (seed,), fit_frac, tag='_greedy')
        df = df[df.candidate != '(base)'].sort_values('d_valid_acc', ascending=False)
        best = df.iloc[0]
        steps.append({'added': best.candidate, 'd_valid_acc': best.d_valid_acc, 'd_valid_lo': best.d_valid_lo,
                      'valid_acc': best.valid_acc, 'stress_acc': best.stress_acc})
        if best.d_valid_acc < min_gain or best.d_valid_lo <= 0:
            steps[-1]['added'] = f'(stop: best was {best.candidate})'
            break
        chosen.append(best.candidate); pool.remove(best.candidate)
        print(f'[greedy] + {best.candidate} → valid {best.valid_acc:.4f}', flush=True)
    pd.DataFrame(steps).to_csv(f'{lab_dir}/screen_greedy.csv', index=False)
    return chosen, pd.DataFrame(steps)
