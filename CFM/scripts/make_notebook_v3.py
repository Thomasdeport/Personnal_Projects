"""Generate CFM_V3.ipynb at the repo root (python scripts/make_notebook_v3.py)."""
from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# CFM — V3 : tous les leviers, une différence à la fois

Point de départ : **0,59999** au LB (blend `sig2_depthaug` + `signature_no_bias`, équilibré).

| § | Levier | Règle de décision, écrite avant les résultats |
|---|---|---|
| 1 | **Validation par grappes de régime** : `valid` réserve des grappes entières (k-means par titre sur profondeur, lots, ticks, venues) pour approcher des jours non vus | — (nouveau protocole, nouveau `lab_v3`) |
| 2 | **Modèle** : 6 configs × 2 seeds (ancre, sans profondeur, σ=0,5, EMA, 60 époques, grand modèle) | score = moyenne(valid-grappes équilibrée, stress équilibré), moyenné sur les seeds |
| 3 | **Ensemble** : 2 meilleures configs × 3 seeds, refit | — |
| 4 | **Alignement de profondeur au test** : on multiplie les tailles du test par γ pour ramener sa profondeur médiane à celle du train | adopté si +0,3 pt en stress équilibré (γ estimé sans labels sur le stress) |
| 5 | **Vote entre voisins** (fenêtres, embeddings, combinaison) | adopté si +0,5 pt sur valid ∪ stress équilibré |
| 6 | Soumissions : A (ensemble équilibré + alignement si adopté) ; B (A + voisins si adopté) | — |

Tous les tableaux de décision sont écrits en CSV dans le lab et inclus dans le zip final.
⚠️ Le sens du stress, l'équilibrage, l'alignement et le vote entre voisins utilisent X_test sans labels (transductif).
""")
code("""
import os, sys, json, subprocess
from pathlib import Path
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')

CODE_DATASET = None   # ex. 'ton-username/cfm-signature-lab'

def find_repo():
    if CODE_DATASET:
        import kagglehub
        path = Path(kagglehub.dataset_download(CODE_DATASET, force_download=True))
        hits = [p.parent.parent for p in path.glob('**/cfm/__init__.py')]
        if not hits:
            raise FileNotFoundError(f'{CODE_DATASET} ne contient pas cfm/')
        return hits[0]
    for p in [Path('.'), Path('..'), Path('/kaggle/working/cfm-signature-lab')]:
        if (p / 'cfm' / '__init__.py').exists():
            return p.resolve()
    for p in Path('/kaggle/input').glob('**/cfm/__init__.py'):
        return p.parent.parent
    raise FileNotFoundError('Code introuvable : renseigner CODE_DATASET ou attacher le Dataset du code')

REPO = find_repo(); sys.path.insert(0, str(REPO))
DEMO = False
SEEDS = [0, 1]
EXTRA_SEEDS = [2]
CONFIGS = {
    'v3_base':    'ancre : sig2_depthaug (σ=0,3), protocole V3',
    'v3_nodepth': 'sans augmentation de profondeur',
    'v3_depth05': 'augmentation de profondeur σ=0,5',
    'v3_ema':     'moyenne exponentielle des poids (0,999)',
    'v3_long':    '60 époques au lieu de 40',
    'v3_big':     'd=128, 3 convolutions (681k paramètres)',
}
N_FINAL_CONFIGS = 2
GAMMA_MULTS = [0.0, 0.5, 1.0, 1.5, 2.0]
MIN_ALIGN_GAIN, MIN_SMOOTH_GAIN = 0.003, 0.005

import numpy as np, pandas as pd
from cfm import config, io, split, blend, plots, proxies, transductive as T
from cfm.__main__ import prepare
from cfm.neural.train import fit, refit, embed, predict_shifted
from cfm.registry import window_matrix
from cfm.metrics import scores, paired_bootstrap
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)

LAB = 'demo_lab3' if DEMO else '/kaggle/working/lab_v3'
DEMO_SET = ['neural.d_model=32', 'neural.heads=2', 'neural.train.epochs=3', 'neural.train.patience=3',
            'neural.train.batch_size=64', 'neural.train.amp=false', 'split.clusters_per_class=4'] if DEMO else []

def load_cfg(name='v3_base', extra=()):
    over = [f'lab_dir={LAB}'] + (['device=cpu'] + DEMO_SET if DEMO else []) + list(extra)
    return config.load(REPO / 'configs' / f'{name}.json', over)

def save(df, name):
    df.to_csv(Path(LAB) / f'v3_{name}.csv'); return df

cfg = load_cfg()
print('repo', REPO, '| lab', LAB, '| device', cfg['device'])
print('code', next(l for l in (REPO / 'CHANGELOG.md').read_text().splitlines() if l.startswith('## ')))
""")
md("## 1. Préparation et validation par grappes de régime")
code("""
files = None
if DEMO:
    from cfm.synthetic import create
    files = create('demo_data3')
sp = prepare(cfg, files)
info = json.loads((Path(LAB) / 'split.json').read_text()); print(json.dumps(info, indent=1))
y_all = np.asarray(io.load(LAB, 'train')['y']).astype(int)
# Contrôle : les fenêtres de valid sont-elles « loin » du fit en régime ? (distance au plus proche voisin du fit)
Z = T.standardise(np.asarray(window_matrix(['levels', 'lots', 'ticks', 'cat_freq'], LAB, 'train', cfg['lot'])[0]))
rng = np.random.default_rng(0)
sample = lambda idx: rng.choice(idx, min(3000, len(idx)), replace=False)
fit_s = sample(sp['fit'])
def nn_dist(rows):
    import torch
    A = torch.tensor(Z[rows]); B = torch.tensor(Z[fit_s])
    d = torch.cdist(A, B); d[torch.isclose(d, torch.zeros_like(d))] = float('inf')
    return float(d.min(1).values.median())
dist = pd.Series({p: nn_dist(sample(sp[p])) for p in ['fit', 'valid', 'stress', 'audit']}, name='distance médiane au fit')
save(dist.to_frame(), 'split_distance'); dist
""")
md("## 2. Banc de modèles : 6 configurations × 2 seeds")
code("""
def bal(run, part):
    z = np.load(Path(LAB) / 'runs' / run / f'{part}.npz')
    return scores(z['y'], blend.sinkhorn_balance(z['p']))['acc']

results = {}
for name in CONFIGS:
    for seed in SEEDS:
        run = f'{name}_s{seed}'
        results[run] = fit(LAB, load_cfg(name, [f'neural.train.seed={seed}']), run, note=f'V3 : {CONFIGS[name]}')
res = pd.DataFrame(results.values()).set_index('name')
res['config'] = [r.rsplit('_s', 1)[0] for r in res.index]
res['valid_bal'] = [bal(r, 'valid') for r in res.index]
res['stress_bal'] = [bal(r, 'stress') for r in res.index]
res['score'] = (res.valid_bal + res.stress_bal) / 2
save(res[['config', 'params', 'best_epoch', 'minutes', 'valid_acc', 'valid_bal', 'stress_acc', 'stress_bal', 'score']], 'runs')
summary = res.groupby('config')[['valid_bal', 'stress_bal', 'score', 'params', 'minutes']].agg(['mean', 'std'])
summary[('Δscore_vs_base', 'pts')] = (summary[('score', 'mean')] - summary.loc['v3_base', ('score', 'mean')]) * 100
summary = summary.sort_values(('score', 'mean'), ascending=False)
save(summary, 'summary'); summary.round(4)
""")
code("""
plots.histories(LAB, [f'{c}_s{SEEDS[0]}' for c in CONFIGS])
prox = proxies.table(LAB, list(results)); save(prox, 'proxies'); prox.round(3)
""")
md("## 3. Ensemble : 2 meilleures configs × 3 seeds, refit")
code("""
finals = list(summary.index[:N_FINAL_CONFIGS])
for c in finals:
    for seed in EXTRA_SEEDS:
        run = f'{c}_s{seed}'
        results[run] = fit(LAB, load_cfg(c, [f'neural.train.seed={seed}']), run, note='V3 : seed finaliste')
MEMBERS = [f'{c}_s{s}' for c in finals for s in SEEDS + EXTRA_SEEDS]
print('membres :', MEMBERS)
cfg_of = lambda m: load_cfg(m.rsplit('_s', 1)[0], [f"neural.train.seed={m.rsplit('_s', 1)[1]}"])
for m in MEMBERS:
    refit(LAB, cfg_of(m), m, 'epochs')
table, w, agree, oracle = blend.report(LAB, MEMBERS)
save(table, 'blend'); plots.agreement(agree, LAB); table
""")
md("""
## 4. Alignement de profondeur au test
Le test est moins profond. On multiplie les tailles du carnet par γ = exp(m·Δ), avec
Δ = médiane(fit) − médiane(cible) du log-profondeur. Δ est estimé **sans labels**.
On choisit le multiplicateur m sur le stress (qui est lui aussi moins profond), puis on l'applique au test avec son propre Δ.
""")
code("""
lv, names, _ = window_matrix(['levels'], LAB, 'train', cfg['lot']); j = names.index('levels:log_depth_q50')
lvt, _, _ = window_matrix(['levels'], LAB, 'test', cfg['lot'])
med = lambda x: float(np.nanmedian(x))
delta_stress = med(lv[sp['fit'], j]) - med(lv[sp['stress'], j])
delta_test = med(lv[:, j]) - med(lvt[:, j])
print(f'Δ stress = {delta_stress:.3f}  |  Δ test = {delta_test:.3f}  (log-profondeur, sans labels)')
ys = y_all[sp['stress']]
Ps = np.mean([predict_shifted(LAB, cfg_of(m), m, 'dev', 'stress', [g * delta_stress for g in GAMMA_MULTS]) for m in MEMBERS], 0)
align = pd.DataFrame({'mult': GAMMA_MULTS, 'gamma': np.exp(np.array(GAMMA_MULTS) * delta_stress),
                      'stress_acc': [scores(ys, P)['acc'] for P in Ps],
                      'stress_bal': [scores(ys, blend.sinkhorn_balance(P))['acc'] for P in Ps]})
save(align, 'depth_alignment')
best_m = float(align.loc[align.stress_bal.idxmax(), 'mult'])
USE_ALIGN = best_m != 0 and align.stress_bal.max() >= align.loc[align.mult == 0, 'stress_bal'].iloc[0] + MIN_ALIGN_GAIN
print(f"meilleur multiplicateur {best_m} → alignement {'ADOPTÉ' if USE_ALIGN else 'rejeté'}")
align.round(4)
""")
code("""
g_test = best_m * delta_test if USE_ALIGN else 0.0
P_test = np.mean([predict_shifted(LAB, cfg_of(m), m, 'refit', 'test', [g_test])[0] for m in MEMBERS], 0)
ids = np.asarray(io.load(LAB, 'test')['ids']); classes = np.asarray(io.meta(LAB)['classes'])
print(f'γ test = {np.exp(g_test):.3f}')
""")
md("## 5. Vote entre voisins")
code("""
pool = np.r_[sp['valid'], sp['stress']]; y_pool = y_all[pool]
P_pool = np.mean([np.r_[np.load(Path(LAB) / 'runs' / m / 'valid.npz')['p'], np.load(Path(LAB) / 'runs' / m / 'stress.npz')['p']]
                  for m in MEMBERS], 0)
for m in MEMBERS:
    embed(LAB, cfg_of(m), m, 'dev', ('valid', 'stress')); embed(LAB, cfg_of(m), m, 'refit', ('test',))
Zv, _ = T.emb_rep(LAB, MEMBERS, 'dev', 'valid'); Zs, _ = T.emb_rep(LAB, MEMBERS, 'dev', 'stress')
Zw = T.window_rep(LAB, cfg, 'train', pool); Ze = T.standardise(np.r_[Zv, Zs])
reps = {'window': Zw, 'embedding': Ze, 'combo': np.c_[Zw, Ze], 'probs': np.sqrt(P_pool)}
pur = pd.DataFrame({k: T.purity(Z, y_pool) for k, Z in reps.items()}).T
save(pur, 'neighbour_purity'); display(pur.round(3))
g = T.grid(P_pool, y_pool, reps); save(g, 'neighbour_grid')
best, none = g.iloc[0], g[g.rep == '(none)'].iloc[0]
USE_SMOOTH = best.rep != '(none)' and best.acc_balanced >= none.acc_balanced + MIN_SMOOTH_GAIN
print(f"sans lissage {none.acc_balanced:.4f} | meilleur {best.rep} k={best.k} α={best.alpha} it={best.iters} {best.acc_balanced:.4f}"
      f" → lissage {'ADOPTÉ' if USE_SMOOTH else 'rejeté'}")
g.head(10).round(4)
""")
md("## 6. Soumissions")
code("""
def write(P, name, note):
    path = Path(LAB) / f'{name}.csv'
    pd.DataFrame({'obs_id': ids, 'eqt_code_cat': classes[P.argmax(1)]}).to_csv(path, index=False)
    np.savez(Path(LAB) / f'{name}_probs.npz', obs_ids=ids, p=P)
    print(name, '←', note); return path

A = blend.sinkhorn_balance(P_test)
write(A, 'submission_v3_A', f'{len(MEMBERS)} modèles, γ={np.exp(g_test):.3f}, équilibré')
decisions = {'finals': finals, 'members': MEMBERS, 'align': bool(USE_ALIGN), 'align_mult': best_m, 'gamma_test': float(np.exp(g_test)),
             'smoothing': bool(USE_SMOOTH)}
if USE_SMOOTH:
    Zw_t = T.window_rep(LAB, cfg, 'test', np.arange(len(ids)))
    Ze_t, eid = T.emb_rep(LAB, MEMBERS, 'refit', 'test'); assert np.array_equal(eid, ids); Ze_t = T.standardise(Ze_t)
    Zt = {'window': Zw_t, 'embedding': Ze_t, 'combo': np.c_[Zw_t, Ze_t], 'probs': np.sqrt(P_test)}[best.rep]
    Bq, kt = T.apply(P_test, Zt, int(best.k), float(best.alpha), int(best.iters), k_scale=4)
    write(Bq, 'submission_v3_B', f'A + voisins {best.rep} k_test={kt}')
    decisions.update({'smooth_rep': best.rep, 'k_test': kt, 'B_changed_vs_A': float((Bq.argmax(1) != A.argmax(1)).mean())})
(Path(LAB) / 'v3_decisions.json').write_text(json.dumps(decisions, indent=2)); decisions
""")
code("""
import zipfile
dest = Path(LAB).parent / ('demo_results_v3.zip' if DEMO else 'lab_results_v3.zip')
with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in Path(LAB).rglob('*'):
        rel = p.relative_to(LAB)
        if p.is_file() and not {'raw', 'features'} & set(rel.parts) and p.suffix != '.pt' and not p.name.startswith(('emb_', 'shift_')):
            z.write(p, rel)
print(dest, round(dest.stat().st_size / 1e6, 1), 'Mo')
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
out = Path(__file__).resolve().parents[1] / 'CFM_V3.ipynb'
nbf.write(nb, out)
print(out)
