"""Generate CFM_V4.ipynb at the repo root (python scripts/make_notebook_v4.py)."""
from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# CFM — V4 : pseudo-étiquetage, entraînement long, réglage du vote entre voisins

Point de départ : **0,6351** au LB (V3 B : 6 modèles, alignement de profondeur, équilibrage, vote entre voisins).

| § | Levier | Règle, écrite avant les résultats |
|---|---|---|
| 1 | Préparation : réutilise `lab_v3` s'il existe (mêmes features, même split par grappes) | — |
| 2 | **Pseudo-étiquetage** : fenêtres test où le professeur V3 B (lissé) et V3 A (brut) sont d'accord, quota équilibré de 40 % par titre | sélection fixe, pas de réglage sur le LB |
| 3 | **Élèves** `v4_student` (2 seeds) : `v3_long` + pseudo-labels (poids 0,5). **Entraînement long** `v4_xlong` (3 seeds, 90 époques). Ancre `v3_long` (2 seeds) | rapporté ; ensemble final fixé d'avance |
| 4 | Refit des 5 membres : `v4_xlong` ×3 + `v4_student` ×2 | — |
| 5 | Alignement de profondeur (comme V3) | adopté si +0,3 pt en stress équilibré |
| 6 | Vote entre voisins : grille sur valid ∪ stress, puis **trois valeurs de k au test** | — |
| 7 | Soumissions : C_k20 / C_k40 / C_k60 (5 modèles) et D_k40 (xlong seul, isole l'effet des pseudo-labels) | à soumettre dans l'ordre C_k40, D_k40, puis la meilleure variante de k |

⚠️ Pseudo-étiquetage, alignement, équilibrage et voisins utilisent X_test sans labels (transductif). Vérifier le règlement.
""")
code("""
import os, sys, json, hashlib
from pathlib import Path
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')

CODE_DATASET = None   # ex. 'ton-username/cfm-signature-lab'
# Professeur V3 : attacher lab_results_v3.zip comme dataset (ou teacher/ via kaggle_upload.py, ou lab_v3 encore présent)

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
PSEUDO_FRAC, PSEUDO_WEIGHT = 0.4, 0.5
XLONG_SEEDS, STUDENT_SEEDS, ANCHOR_SEEDS = [0, 1, 2], [0, 1], [0, 1]
GAMMA_MULTS = [0.0, 0.5, 1.0, 1.5]
MIN_ALIGN_GAIN = 0.003
K_TEST = [20, 40, 60]

import numpy as np, pandas as pd
from cfm import config, io, blend, plots, transductive as T
from cfm.__main__ import prepare
from cfm.neural.train import fit, refit, embed, predict_shifted
from cfm.registry import window_matrix
from cfm.metrics import scores
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)

LAB = 'demo_lab4' if DEMO else '/kaggle/working/lab_v3'
DEMO_SET = ['neural.d_model=32', 'neural.heads=2', 'neural.train.epochs=3', 'neural.train.patience=3',
            'neural.train.batch_size=64', 'neural.train.amp=false', 'split.clusters_per_class=4'] if DEMO else []
PSEUDO_SHA = None

def load_cfg(name='v3_base', extra=()):
    over = [f'lab_dir={LAB}'] + (['device=cpu'] + DEMO_SET if DEMO else []) + list(extra)
    c = config.load(REPO / 'configs' / f'{name}.json', over)
    if c['neural'].get('pseudo') is not None:
        c['neural']['pseudo']['sha'] = PSEUDO_SHA
    return c

cfg_of = lambda m: load_cfg(m.rsplit('_s', 1)[0], [f"neural.train.seed={m.rsplit('_s', 1)[1]}"])
def save(df, name):
    df.to_csv(Path(LAB) / f'v4_{name}.csv'); return df

cfg = load_cfg()
print('repo', REPO, '| lab', LAB, '| device', cfg['device'])
print('code', next(l for l in (REPO / 'CHANGELOG.md').read_text().splitlines() if l.startswith('## ')))
""")
md("## 1. Préparation")
code("""
files = None
if DEMO:
    from cfm.synthetic import create
    files = create('demo_data4')
sp = prepare(cfg, files)
y_all = np.asarray(io.load(LAB, 'train')['y']).astype(int)
ids = np.asarray(io.load(LAB, 'test')['ids']); classes = np.asarray(io.meta(LAB)['classes'])
print(json.loads((Path(LAB) / 'split.json').read_text())['sizes'])
""")
md("## 2. Pseudo-labels issus du professeur V3")
code("""
if DEMO:   # pas de professeur réel : on fabrique un professeur bruité pour vérifier la mécanique
    rng = np.random.default_rng(0); A_p = rng.dirichlet(np.ones(len(classes)), len(ids)); B_p = (A_p + rng.dirichlet(np.ones(len(classes)), len(ids))) / 2
else:
    # Pas dans git (prédictions test privées) : cherchés dans teacher/, dans lab_v3, puis dans tout dataset attaché
    # (le plus simple : attacher lab_results_v3.zip comme dataset Kaggle).
    A, B = np.load(T.find_teacher('A', REPO, LAB)), np.load(T.find_teacher('B', REPO, LAB))
    assert np.array_equal(A['obs_ids'], ids) and np.array_equal(B['obs_ids'], ids), 'professeur non aligné sur le test'
    A_p, B_p = A['p'], B['p']
sel, ylab, tab = T.select_pseudo(B_p, A_p, PSEUDO_FRAC)
path = Path(LAB) / 'pseudo_v4.npz'
np.savez(path, obs_ids=ids, idx=sel, y=ylab)
PSEUDO_SHA = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
summary_ps = {'selected': int(len(sel)), 'share_of_test': float(len(sel) / len(ids)),
              'agreement_A_B': float((A_p.argmax(1) == B_p.argmax(1)).mean()),
              'min_conf_selected': float(B_p[sel].max(1).min()), 'mean_conf_selected': float(B_p[sel].max(1).mean()),
              'sha': PSEUDO_SHA}
save(tab, 'pseudo_per_class'); print(json.dumps(summary_ps, indent=1))
""")
md("## 3. Modèles : élèves, entraînement long, ancre")
code("""
def bal(run, part):
    z = np.load(Path(LAB) / 'runs' / run / f'{part}.npz')
    return scores(z['y'], blend.sinkhorn_balance(z['p']))['acc']

plan = [('v3_long', s) for s in ANCHOR_SEEDS] + [('v4_student', s) for s in STUDENT_SEEDS] + [('v4_xlong', s) for s in XLONG_SEEDS]
results = {}
for name, seed in plan:
    run = f'{name}_s{seed}'
    results[run] = fit(LAB, load_cfg(name, [f'neural.train.seed={seed}']), run, note=f'V4 : {name}')
res = pd.DataFrame(results.values()).set_index('name')
res['config'] = [r.rsplit('_s', 1)[0] for r in res.index]
res['valid_bal'] = [bal(r, 'valid') for r in res.index]
res['stress_bal'] = [bal(r, 'stress') for r in res.index]
res['score'] = (res.valid_bal + res.stress_bal) / 2
# indicateur sans labels : accord avec le professeur sur le test (les élèves l'ont vu en partie : biais attendu)
res['test_agree_teacher'] = [(np.load(Path(LAB) / 'runs' / r / 'test_dev.npz')['p'].argmax(1) == B_p.argmax(1)).mean() for r in res.index]
save(res[['config', 'best_epoch', 'minutes', 'valid_bal', 'stress_bal', 'score', 'test_agree_teacher']], 'runs')
save(res.groupby('config')[['valid_bal', 'stress_bal', 'score', 'test_agree_teacher', 'minutes']].agg(['mean', 'std']), 'summary').round(4)
""")
code("""
plots.histories(LAB, [f'v3_long_s{ANCHOR_SEEDS[0]}', f'v4_student_s{STUDENT_SEEDS[0]}', f'v4_xlong_s{XLONG_SEEDS[0]}'])
""")
md("## 4. Refit des membres (ensemble fixé d'avance)")
code("""
XL = [f'v4_xlong_s{s}' for s in XLONG_SEEDS]; ST = [f'v4_student_s{s}' for s in STUDENT_SEEDS]
MEMBERS = XL + ST
for m in MEMBERS:
    refit(LAB, cfg_of(m), m, 'epochs')
table, w, agree, oracle = blend.report(LAB, MEMBERS); save(table, 'blend'); plots.agreement(agree, LAB); table
""")
md("## 5. Alignement de profondeur")
code("""
lv, names, _ = window_matrix(['levels'], LAB, 'train', cfg['lot']); j = names.index('levels:log_depth_q50')
lvt, _, _ = window_matrix(['levels'], LAB, 'test', cfg['lot'])
med = lambda x: float(np.nanmedian(x))
d_stress = med(lv[sp['fit'], j]) - med(lv[sp['stress'], j]); d_test = med(lv[:, j]) - med(lvt[:, j])
ys = y_all[sp['stress']]
Ps = np.mean([predict_shifted(LAB, cfg_of(m), m, 'dev', 'stress', [g * d_stress for g in GAMMA_MULTS]) for m in MEMBERS], 0)
align = save(pd.DataFrame({'mult': GAMMA_MULTS, 'stress_bal': [scores(ys, blend.sinkhorn_balance(P))['acc'] for P in Ps]}), 'depth_alignment')
best_m = float(align.loc[align.stress_bal.idxmax(), 'mult'])
USE_ALIGN = best_m != 0 and align.stress_bal.max() >= align.loc[align.mult == 0, 'stress_bal'].iloc[0] + MIN_ALIGN_GAIN
g_test = best_m * d_test if USE_ALIGN else 0.0
print(f"Δ test {d_test:.3f} → γ test {np.exp(g_test):.3f} ({'adopté' if USE_ALIGN else 'rejeté'})"); align.round(4)
""")
code("""
Pt = {m: predict_shifted(LAB, cfg_of(m), m, 'refit', 'test', [g_test])[0] for m in MEMBERS}
P_C = np.mean([Pt[m] for m in MEMBERS], 0)     # xlong + élèves
P_D = np.mean([Pt[m] for m in XL], 0)          # xlong seul
""")
md("## 6. Vote entre voisins")
code("""
pool = np.r_[sp['valid'], sp['stress']]; y_pool = y_all[pool]
P_pool = np.mean([np.r_[np.load(Path(LAB) / 'runs' / m / 'valid.npz')['p'], np.load(Path(LAB) / 'runs' / m / 'stress.npz')['p']]
                  for m in MEMBERS], 0)
for m in MEMBERS:
    embed(LAB, cfg_of(m), m, 'dev', ('valid', 'stress')); embed(LAB, cfg_of(m), m, 'refit', ('test',))
def combo(members, weights, part_rows=None):
    if weights == 'dev':
        Zv, _ = T.emb_rep(LAB, members, 'dev', 'valid'); Zs, _ = T.emb_rep(LAB, members, 'dev', 'stress')
        return np.c_[T.window_rep(LAB, cfg, 'train', pool), T.standardise(np.r_[Zv, Zs])]
    Zt, eid = T.emb_rep(LAB, members, 'refit', 'test'); assert np.array_equal(eid, ids)
    return np.c_[T.window_rep(LAB, cfg, 'test', np.arange(len(ids))), T.standardise(Zt)]
g = save(T.grid(P_pool, y_pool, {'combo': combo(MEMBERS, 'dev')}), 'neighbour_grid')
best = g[g.rep == 'combo'].iloc[0]; none = g[g.rep == '(none)'].iloc[0]
print(f"pool : sans {none.acc_balanced:.4f} | combo k={best.k} α={best.alpha} it={best.iters} → {best.acc_balanced:.4f}")
g.head(8).round(4)
""")
md("## 7. Soumissions")
code("""
def write(P, name, note):
    path = Path(LAB) / f'{name}.csv'
    pd.DataFrame({'obs_id': ids, 'eqt_code_cat': classes[P.argmax(1)]}).to_csv(path, index=False)
    np.savez(Path(LAB) / f'{name}_probs.npz', obs_ids=ids, p=P); print(f'{name:22s} ← {note}'); return P

Zc, Zd = combo(MEMBERS, 'refit'), combo(XL, 'refit')
out = {}
for kt in K_TEST:
    out[f'C_k{kt}'] = write(T.sinkhorn_balance(T.smooth(P_C, T.knn(Zc, kt), float(best.alpha), int(best.iters))),
                            f'submission_v4_C_k{kt}', f'xlong×{len(XL)} + élèves×{len(ST)}, voisins k={kt}')
out['D_k40'] = write(T.sinkhorn_balance(T.smooth(P_D, T.knn(Zd, 40), float(best.alpha), int(best.iters))),
                     'submission_v4_D_k40', f'xlong×{len(XL)} seul, voisins k=40')
teacher = B_p.argmax(1)
diff = pd.DataFrame({k: {'vs_teacher_V3B': float((v.argmax(1) != teacher).mean()),
                         'vs_C_k40': float((v.argmax(1) != out['C_k40'].argmax(1)).mean())} for k, v in out.items()}).T
save(diff, 'submission_diffs')
decisions = {'pseudo': summary_ps, 'members': MEMBERS, 'align': bool(USE_ALIGN), 'gamma_test': float(np.exp(g_test)),
             'neighbours': {'alpha': float(best.alpha), 'iters': int(best.iters), 'k_pool': int(best.k), 'k_test': K_TEST},
             'submit_order': ['submission_v4_C_k40', 'submission_v4_D_k40', 'meilleure variante de k']}
(Path(LAB) / 'v4_decisions.json').write_text(json.dumps(decisions, indent=2)); diff.round(3)
""")
code("""
import zipfile
dest = Path(LAB).parent / ('demo_results_v4.zip' if DEMO else 'lab_results_v4.zip')
with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in Path(LAB).rglob('*'):
        rel = p.relative_to(LAB)
        if p.is_file() and not {'raw', 'features'} & set(rel.parts) and p.suffix != '.pt' and not p.name.startswith(('emb_', 'shift_')):
            z.write(p, rel)
print(dest, round(dest.stat().st_size / 1e6, 1), 'Mo')
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
out = Path(__file__).resolve().parents[1] / 'CFM_V4.ipynb'
nbf.write(nb, out)
print(out)
