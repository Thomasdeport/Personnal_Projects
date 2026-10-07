"""Contract tests on synthetic data. `python tests/test_lab.py` (no pytest needed) or `pytest -q tests`.

They validate the SOFTWARE (alignment, leakage guards, invariances, caching, resume), never performance.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # macOS: torch + xgboost load two OpenMP runtimes
import json
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cfm import blend, config, io, registry, split, synthetic  # noqa: E402
from cfm.__main__ import prepare  # noqa: E402
from cfm.features.common import Raw  # noqa: E402

_STATE = {}


def lab():
    """Build one small synthetic lab per session."""
    if 'cfg' not in _STATE:
        tmp = Path(tempfile.mkdtemp(prefix='cfm_test_'))
        files = synthetic.create(tmp / 'data', n_per_class=12, n_test_per_class=4, seed=1)
        cfg = config.load(ROOT / 'configs' / 'demo.json', [f'lab_dir={tmp / "lab"}'])
        prepare(cfg, files)
        _STATE.update(cfg=cfg, tmp=tmp, files=files)
    return _STATE['cfg'], _STATE['files']


def test_labels_aligned_with_obs_id():
    cfg, files = lab()
    raw = io.load(cfg['lab_dir'], 'train')
    truth = pd.read_csv(files['target']).set_index('obs_id').eqt_code_cat
    classes = np.asarray(io.meta(cfg['lab_dir'])['classes'])
    assert (classes[np.asarray(raw['y'])] == truth.loc[np.asarray(raw['ids'])].to_numpy()).all()
    first = pd.read_csv(files['train'], nrows=100)
    assert raw['ids'][0] == first.obs_id.iloc[0]
    assert np.allclose(raw['num'][0, :, 5], first.flux.to_numpy())


def test_rejects_non_contiguous_windows():
    cfg, files = lab()
    x = pd.read_csv(files['train'], nrows=200)
    x = pd.concat([x.iloc[:50], x.iloc[100:150], x.iloc[50:100], x.iloc[150:]])
    d = _STATE['tmp'] / 'bad'; d.mkdir(exist_ok=True)
    x.to_csv(d / 'X_train.csv', index=False); x.to_csv(d / 'X_test.csv', index=False)
    pd.DataFrame({'obs_id': x.obs_id.unique(), 'eqt_code_cat': [0, 1]}).to_csv(d / 'y_train.csv', index=False)
    try:
        io.build({'train': d / 'X_train.csv', 'test': d / 'X_test.csv', 'target': d / 'y_train.csv'}, d / 'lab')
    except ValueError as e:
        assert 'contiguous' in str(e)
    else:
        raise AssertionError('non-contiguous windows were accepted')


def test_order_rank_keeps_only_equalities():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 30, (50, 100)).astype(float)
    ids[0, :5] = np.nan
    relabel = ids.copy()
    for i in range(len(ids)):                     # bijective renumbering per window
        perm = rng.permutation(10_000)[:30] * 7 + 3
        ok = np.isfinite(ids[i])
        relabel[i, ok] = perm[ids[i, ok].astype(int)]
    a, b = io.local_order_rank(ids), io.local_order_rank(relabel)
    assert (a == b).all()
    assert len(set(a[0, :5])) == 5                # missing ids never match each other
    for i in range(5):                            # rank = order of first appearance
        seen = {}
        for j, v in enumerate(ids[i]):
            key = ('nan', j) if np.isnan(v) else v
            seen.setdefault(key, len(seen))
            assert a[i, j] == seen[key]


def test_order_helpers_match_naive_loop():
    rng = np.random.default_rng(1)
    oid = io.local_order_rank(rng.integers(0, 15, (20, 100)))
    action = rng.integers(1, 4, (20, 100))
    arrays = {'num': np.zeros((20, 100, 6)), 'cat': np.stack([np.ones((20, 100)), action, np.ones((20, 100)),
                                                              np.ones((20, 100))], -1), 'oid': oid}
    r = Raw(arrays, {'tick': .01, 'lot': 100, 'n_venue': 6})
    for i in range(20):
        last, count, firsta = {}, {}, {}
        total = pd.Series(oid[i]).value_counts()
        for j, o in enumerate(oid[i]):
            assert r.occ[i, j] == count.get(o, 0)
            assert r.prev_action[i, j] == (action[i, last[o]] if o in last else 0)
            assert r.gap[i, j] == (j - last[o] if o in last else 0)
            firsta.setdefault(o, action[i, j])
            assert r.first_action[i, j] == firsta[o]
            assert r.order_count[i, j] == total[o]
            count[o] = count.get(o, 0) + 1; last[o] = j


def test_tick_units_on_handcrafted_window():
    n = 1
    bid = np.zeros((n, 100)); ask = np.full((n, 100), .01)
    ask[:, 50:] = .02; bid[:, 80:] = .01; ask[:, 80:] = .02           # spread 1 → 2 → 1 tick
    side = np.ones((n, 100)); side[:, ::2] = 2                         # alternate A / B
    price = np.where(side == 2, bid, ask)                              # every event at its best
    price[:, 1] = ask[:, 1] + .02                                      # one ask event 2 ticks deeper
    num = np.stack([price, bid, ask, np.full((n, 100), 300.), np.full((n, 100), 100.), np.full((n, 100), 100.)], -1)
    cat = np.stack([np.ones((n, 100)), np.ones((n, 100)), side, np.ones((n, 100))], -1)
    r = Raw({'num': num, 'cat': cat, 'oid': np.zeros((n, 100), np.int8)}, {'tick': .01, 'lot': 100, 'n_venue': 6})
    from cfm.features.window import ticks, position, lots
    t, p, lt = ticks(r), position(r), lots(r)
    assert np.isclose(t['spread_1'][0], .7) and np.isclose(t['spread_2'][0], .3)   # 50+20 vs 30 events
    assert np.isclose(p['pos_best'][0], .99) and np.isclose(p['pos_d2'][0], .01)
    assert np.isclose(lt['flux_exactly_1lot'][0], 1)
    assert np.isclose(r.dmid_h[0, 50], 1) and np.isclose(r.dmid_h[0, 80], 1)   # half-tick moves


def test_relative_blocks_invariant_to_size_scaling_levels_are_not():
    cfg, _ = lab()
    raw = io.load(cfg['lab_dir'], 'train')
    arr = {k: np.asarray(raw[k][:30]) for k in ['num', 'cat', 'oid']}
    scaled = {**arr, 'num': arr['num'].copy()}
    scaled['num'][..., 3:] *= 3.0
    ctx = registry.context(cfg['lab_dir'], cfg['lot'])
    from cfm.features.window import base_relative, levels
    a, b = base_relative(Raw(arr, ctx)), base_relative(Raw(scaled, ctx))
    for k in a:
        assert np.allclose(a[k], b[k], atol=1e-6, equal_nan=True), k
    assert not np.allclose(levels(Raw(arr, ctx))['log_depth_q50'], levels(Raw(scaled, ctx))['log_depth_q50'])


def test_feature_cache_is_keyed_on_context():
    cfg, _ = lab()
    k1 = registry.block_key('lots', cfg['lab_dir'], 100.)
    k2 = registry.block_key('lots', cfg['lab_dir'], 50.)
    assert k1 != k2
    x1, _ = registry.compute('lots', cfg['lab_dir'], 'train', 100.)
    x2, _ = registry.compute('lots', cfg['lab_dir'], 'train', 50.)
    assert not np.allclose(np.nan_to_num(x1), np.nan_to_num(x2))


def test_split_disjoint_and_audit_independent_of_stress():
    cfg, _ = lab()
    sp = split.load(cfg['lab_dir'])
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == len(np.unique(allidx)) == len(io.load(cfg['lab_dir'], 'train')['y'])
    y = np.asarray(io.load(cfg['lab_dir'], 'train')['y'])
    for part in sp.values():
        assert len(np.unique(y[part])) == 24
    other = Path(tempfile.mkdtemp())
    for f in ['raw', 'features']:
        shutil.copytree(Path(cfg['lab_dir']) / f, other / f)
    cfg2 = json.loads(json.dumps(cfg)); cfg2['split']['stress_tail'] = 'high'
    sp2 = split.make(other, cfg2)
    assert np.array_equal(sp['audit'], sp2['audit'])
    assert not np.array_equal(sp['stress'], sp2['stress'])


def test_tabular_end_to_end_and_test_alignment():
    cfg, files = lab()
    from cfm.tabular import run
    res = run(cfg['lab_dir'], cfg, ['base_relative', 'cat_freq', 'levels', 'ticks'], 'xgb', name='t_xgb', refit=True)
    assert 0 <= res['valid_acc'] <= 1
    truth = pd.read_csv(Path(files['train']).parent / 'y_test_TRUTH_synthetic_only.csv').set_index('obs_id').eqt_code_cat
    z = np.load(Path(cfg['lab_dir']) / 'runs' / 't_xgb' / 'test_refit.npz')
    acc = (z['p'].argmax(1) == truth.loc[z['obs_ids']].to_numpy()).mean()
    assert acc > 3 / 24, f'test predictions look misaligned (acc {acc:.3f})'
    path = blend.submission(cfg['lab_dir'], ['t_xgb'])
    sub = pd.read_csv(path)
    assert list(sub.columns) == ['obs_id', 'eqt_code_cat']
    assert np.array_equal(sub.obs_id, io.load(cfg['lab_dir'], 'test')['ids'])


def test_neural_runs_resumes_and_refuses_other_config():
    cfg, _ = lab()
    from cfm.neural.train import fit, refit
    c = json.loads(json.dumps(cfg)); c['neural']['train']['epochs'] = 2
    r1 = fit(cfg['lab_dir'], c, 'nn')
    r2 = fit(cfg['lab_dir'], c, 'nn')                       # finished run: reloaded, identical scores
    assert r1['valid_acc'] == r2['valid_acc'] and r1['valid_logloss'] == r2['valid_logloss']
    c2 = json.loads(json.dumps(c)); c2['neural']['d_model'] = 16
    try:
        fit(cfg['lab_dir'], c2, 'nn')
    except ValueError:
        pass
    else:
        raise AssertionError('a different config resumed an existing run')
    z = np.load(refit(cfg['lab_dir'], c, 'nn', 'steps'))
    assert np.allclose(z['p'].sum(1), 1, atol=1e-4)
    for enc in ['conv', 'gru', 'conv_gru', 'rel']:
        c3 = json.loads(json.dumps(c)); c3['neural']['encoder'] = enc; c3['neural']['train']['epochs'] = 1
        fit(cfg['lab_dir'], c3, f'nn_{enc}')


def test_neural_depth_aug_and_embeddings():
    cfg, _ = lab()
    from cfm.neural.train import embed, fit, refit
    c = config.load(ROOT / 'configs' / 'sig2_depthaug.json', [f'lab_dir={cfg["lab_dir"]}'])
    c['device'] = 'cpu'; c['neural'].update({'d_model': 32, 'heads': 2})
    c['neural']['train'].update({'epochs': 1, 'batch_size': 64, 'amp': False})
    fit(cfg['lab_dir'], c, 'aug')
    refit(cfg['lab_dir'], c, 'aug', 'epochs')
    z = embed(cfg['lab_dir'], c, 'aug', 'dev', ('valid', 'stress'))
    assert z['valid'].shape[1] == 32 + c['neural']['context_dim']
    zt = embed(cfg['lab_dir'], c, 'aug', 'refit', ('test',))
    assert len(zt['test']) == len(io.load(cfg['lab_dir'], 'test')['ids'])
    try:
        embed(cfg['lab_dir'], c, 'aug', 'refit', ('valid',))
    except ValueError:
        pass
    else:
        raise AssertionError('refit weights must not embed labelled partitions')
    c2 = config.load(ROOT / 'configs' / 'sig2_nolevels.json', [f'lab_dir={cfg["lab_dir"]}'])
    c2['device'] = 'cpu'; c2['neural'].update({'d_model': 32, 'heads': 2})
    c2['neural']['train'].update({'epochs': 1, 'batch_size': 64, 'amp': False})
    fit(cfg['lab_dir'], c2, 'nolev')


def test_depth_scale_moves_only_depth_channels():
    import torch
    from cfm.neural.train import Augment
    from cfm.neural.data import Scaler
    rng = np.random.default_rng(0)
    cont = np.log1p(rng.gamma(2, 3, (8, 100, 3))).astype(np.float32)
    ctx = rng.normal(size=(8, 2)).astype(np.float32)
    sc = Scaler(cont, ctx, np.arange(8))
    data = {'cont_names': ['log_bq_lots', 'imbalance', 'log_aq_lots'], 'ctx_names': ['levels:log_depth_q50', 'ticks:spread_1'],
            'ctx_block': np.array(['levels', 'ticks']), 'ctx_level_mask': np.array([True, False])}
    aug = Augment({'aug': {'depth_scale': .5}, 'tokens': []}, data, torch.device('cpu'), sc)
    c0, x0 = torch.tensor(sc.cont(cont)), torch.tensor(sc.ctx(ctx))
    _, c1, x1, _ = aug(torch.zeros(8, 100, 0, dtype=torch.long), c0, x0, None)
    assert torch.equal(c1[..., 1], c0[..., 1]) and torch.equal(x1[:, 1], x0[:, 1])
    assert not torch.allclose(c1[..., 0], c0[..., 0]) and not torch.allclose(x1[:, 0], x0[:, 0])
    # same factor for bid and ask of a window: the raw ratio of sizes is preserved
    raw = lambda c, j: torch.expm1(c[..., j] * float(sc.cs[j]) + float(sc.cm[j]))
    r0 = raw(c0, 0)[:, :5] / raw(c0, 2)[:, :5]; r1 = raw(c1, 0)[:, :5] / raw(c1, 2)[:, :5]
    assert torch.allclose(r0, r1, rtol=1e-3)


def test_knn_smoothing_fixes_isolated_errors():
    from cfm import transductive as T
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(24), 20)                       # 20 windows per "stock-day"
    Z = np.eye(24)[y] * 5 + rng.normal(0, .3, (480, 24))   # same group → close
    P = np.full((480, 24), .01); P[np.arange(480), y] = .5
    wrong = rng.random(480) < .3                            # 30 % confidently wrong
    P[wrong] = .01; P[wrong, (y[wrong] + 1) % 24] = .5
    P /= P.sum(1, keepdims=True)
    nb = T.knn(Z, 10, device='cpu')
    assert not (nb == np.arange(480)[:, None]).any()       # self excluded
    assert (T.smooth(P, nb, .8, 5).argmax(1) == y).mean() > (P.argmax(1) == y).mean() + .15
    assert T.purity(Z, y)['purity@10'] > .95


def test_proxies_are_label_free_and_sane():
    from cfm import proxies
    p = np.full((240, 24), 1 / 24); p[np.arange(240), np.arange(240) % 24] = .9
    p /= p.sum(1, keepdims=True)
    i = proxies.indicators(p)
    assert abs(i['balance_kl']) < 1e-9 and i['max_min_ratio'] == 1


def test_neural_v3_ema_and_depth_alignment():
    cfg, _ = lab()
    from cfm.neural.train import fit, predict_shifted, refit
    c = config.load(ROOT / 'configs' / 'v3_ema.json', [f'lab_dir={cfg["lab_dir"]}'])
    c['split'] = cfg['split']                       # same lab, same split
    c['device'] = 'cpu'; c['neural'].update({'d_model': 32, 'heads': 2})
    c['neural']['train'].update({'epochs': 2, 'batch_size': 64, 'amp': False})
    r1 = fit(cfg['lab_dir'], c, 'ema'); r2 = fit(cfg['lab_dir'], c, 'ema')
    assert r1['valid_acc'] == r2['valid_acc']                   # resume restores the EMA too
    refit(cfg['lab_dir'], c, 'ema', 'epochs')
    assert (Path(cfg['lab_dir']) / 'runs' / 'ema' / 'refit_weights.pt').exists()
    P = predict_shifted(cfg['lab_dir'], c, 'ema', 'dev', 'stress', [0., .3])
    z = np.load(Path(cfg['lab_dir']) / 'runs' / 'ema' / 'stress.npz')
    assert np.allclose(P[0], z['p'], atol=1e-5)                  # γ = 1 reproduces the saved predictions
    assert not np.allclose(P[0], P[1])                           # γ ≠ 1 changes them
    Pt = predict_shifted(cfg['lab_dir'], c, 'ema', 'refit', 'test', [0.])
    assert np.allclose(Pt.sum(-1), 1, atol=1e-4)
    try:
        predict_shifted(cfg['lab_dir'], c, 'ema', 'refit', 'valid', [0.])
    except ValueError:
        pass
    else:
        raise AssertionError('refit weights must not predict labelled partitions')


def test_cluster_validation_split():
    cfg, _ = lab()
    other = Path(tempfile.mkdtemp())
    for f in ['raw', 'features']:
        shutil.copytree(Path(cfg['lab_dir']) / f, other / f)
    c = json.loads(json.dumps(cfg)); c['split'].update({'valid_mode': 'cluster', 'clusters_per_class': 3})
    sp = split.make(other, c)
    ref = split.load(cfg['lab_dir'])
    assert np.array_equal(sp['audit'], ref['audit']) and np.array_equal(sp['stress'], ref['stress'])
    allidx = np.concatenate(list(sp.values()))
    assert len(allidx) == len(np.unique(allidx)) == len(io.load(other, 'train')['y'])
    y = np.asarray(io.load(other, 'train')['y'])
    assert len(np.unique(y[sp['valid']])) == 24 and len(sp['valid']) >= .1 * len(y)


def test_sinkhorn_balances_columns():
    p = np.random.default_rng(0).dirichlet(np.ones(24) * .3, 480)
    q = blend.sinkhorn_balance(p)
    assert np.allclose(q.sum(1), 1)
    assert np.allclose(q.sum(0), 20, rtol=1e-2)


if __name__ == '__main__':
    fails = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith('test_')]
    # macOS: XGBoost then torch in one process can deadlock (two OpenMP runtimes) → torch tests first.
    tests.sort(key=lambda t: 'neural' not in t[0])
    for name, fn in tests:
            try:
                fn(); print('PASS', name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print('FAIL', name, repr(e))
    print(f'{fails} failure(s)')
    sys.exit(1 if fails else 0)
