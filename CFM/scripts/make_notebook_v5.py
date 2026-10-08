"""Generate CFM_V5.ipynb at the repo root (python scripts/make_notebook_v5.py)."""
from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# CFM Signature Lab — V5 : second tour d'auto-apprentissage

## Le problème

Le challenge CFM (ENS Challenge Data n° 146) consiste à reconnaître, parmi **24 titres**, celui qui a produit une fenêtre de
**100 événements** de carnet d'ordres : ajouts, annulations, modifications et transactions, avec la venue, le côté, le prix,
la taille et les quantités au meilleur prix. Le test est tiré d'une **autre période** que le train. Toute la difficulté
est là : un modèle qui reconnaît le *jour* plutôt que le *titre* obtient 0,85 en validation aléatoire mais 0,57 au leaderboard.

## Ce qui a été construit jusqu'ici

| Étape | Idée | LB public |
|---|---|---|
| V2 (point de départ) | Transformer hybride sur des features relatives | 0,5072 |
| R1 | Unités naturelles (ticks, lots), événements comme tokens, SignatureNet (convolutions dilatées + attention + contexte) | 0,5700 |
| R1 équilibré | Équilibrage Sinkhorn des prédictions test (le test est ~équilibré entre titres) | 0,5958 |
| R2 | Augmentation de profondeur (les carnets du test sont moins profonds) | 0,59999 |
| V3 A | Validation par grappes de régimes, 60 époques, EMA, alignement de profondeur au test | 0,6028 |
| V3 B | **Vote entre voisins** : les 20 fenêtres d'un même titre-jour se ressemblent ; on lisse les prédictions sur le graphe des k plus proches voisins | 0,6351 |
| **V4 C** | **Auto-apprentissage, tour 1** : 40 % du test pseudo-étiquetés par V3 B, élèves entraînés dessus | **0,6551** |
| V4 D (contrôle) | V4 C sans les élèves : mêmes post-traitements, 90 époques seules | 0,6174 |

## Ce que la V4 a appris

Le contrôle D isole une seule différence, les élèves. Résultat : **+3,8 pts au LB grâce aux pseudo-labels**, alors que
l'entraînement long seul *perd* 1,8 pt par rapport à V3 B. De plus, C et D sont d'accord à 98,9 % sur les fenêtres
pseudo-étiquetées, mais seulement à 81,5 % sur les 60 % restantes. Les élèves n'ont donc pas recopié le professeur :
ils ont appris quelque chose de la période test et l'ont **étendu aux fenêtres qu'on ne leur avait pas données**.
C'est une adaptation de domaine par auto-apprentissage.

## Les hypothèses de la V5

- **H1, meilleur professeur.** V4 C (0,6551) remplace V3 B (0,6351) comme professeur. Ses pseudo-labels sont plus
  justes, donc les élèves du tour 2 devraient faire mieux que ceux du tour 1.
- **H2, quota plus large.** Le gain se fait sur les fenêtres *non* étiquetées. En passer 60 % au lieu de 40 % donne plus
  de signal sur la période test, au prix de labels un peu moins sûrs.
- **Pas de modèles longs.** `v4_xlong` est abandonné (gain interne qui ne transfère pas). Les élèves gardent le budget de `v3_long` (60 époques).

## Le protocole, écrit avant les résultats

| § | Étape | Règle |
|---|---|---|
| 2 | Pseudo-labels : fenêtres où V4 C (lissé) et la moyenne brute des 5 modèles V4 concordent, puis les plus confiantes par titre prédit, jusqu'à q·N/24 | q = 0,6 (élèves E) et q = 0,4 (contrôle F), même professeur ; la sélection 40 % est incluse dans la sélection 60 % |
| 3 | Élèves `v5_q60` ×3 et `v5_q40` ×3 graines | une seule différence entre les deux : le quota |
| 4 | Refit des 6 élèves sur tous les labels + leurs pseudo-labels | ensemble fixé d'avance |
| 5 | Alignement de profondeur | adopté si +0,3 pt en stress équilibré |
| 6 | Vote entre voisins : grille sur le pool valid ∪ stress, k_test = 40 | — |
| 7 | Soumissions **V5_ALL_k40** (6 élèves), puis **V5_E_k40** (q60) et **V5_F_k40** (q40) | soumettre ALL d'abord ; E contre F mesure l'effet du quota (H2) ; ALL contre V4 C mesure H1 |

**Pronostic, à confronter au LB :** ALL ≈ 0,66–0,67. Si ALL < 0,6551, le second tour n'apporte rien : on s'arrête là.

**Mise en garde.** Les scores valid et stress des élèves sont **contaminés** : leur professeur a été refit sur valid et
stress. Ils servent à vérifier que l'entraînement s'est bien passé, pas à mesurer le gain. Seul le LB tranche.

⚠️ Pseudo-labels, alignement, équilibrage et voisins utilisent X_test sans labels (transductif). Vérifier le règlement.

## Mise en place sur Kaggle

1. Attacher le code (`CODE_DATASET`) et les CSV du challenge, activer le GPU. Si `/kaggle/working/lab_v3` existe, il est
   réutilisé ; sinon le cache est reconstruit (≈ 20 min).
2. Attacher un dataset contenant `v4_C_k40_probs.npz` et `v4_raw_probs.npz` (dossier `teacher/` du repo, non commité),
   puis renseigner `TEACHER_DIR`.
3. Durée : ≈ 4 h à 4 h 30 sur GPU (estimation, non mesurée) — 6 entraînements de 16–19 min, autant de refits, puis les post-traitements.
4. Sorties : `lab_results_v5.zip` (tableaux, décisions, soumissions) et **`figures/v5/`**, en PNG et PDF, avec
   `index.md` qui donne la légende de chaque figure, prêtes pour le rapport.
""")
code("""
import os, sys, json, hashlib
from pathlib import Path
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')

CODE_DATASET = None   # ex. 'ton-username/cfm-signature-lab'
TEACHER_DIR = None    # ex. '/kaggle/input/cfm-teacher-v4' : doit contenir v4_C_k40_probs.npz et v4_raw_probs.npz

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
QUOTAS = {'q60': 0.6, 'q40': 0.4}          # élèves E et contrôle F
PSEUDO_WEIGHT = 0.5
SEEDS = [0, 1, 2]
GAMMA_MULTS = [0.0, 0.5, 1.0, 1.5]
MIN_ALIGN_GAIN = 0.003
K_TEST = 40

import numpy as np, pandas as pd
from cfm import config, io, blend, report as R, transductive as T
from cfm.__main__ import prepare
from cfm.neural.train import fit, refit, embed, predict_shifted
from cfm.registry import window_matrix
from cfm.metrics import scores
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)

LAB = 'demo_lab5' if DEMO else '/kaggle/working/lab_v3'
FIG = Path('demo_figures_v5' if DEMO else '/kaggle/working/figures/v5')
DEMO_SET = ['neural.d_model=32', 'neural.heads=2', 'neural.train.epochs=3', 'neural.train.patience=3',
            'neural.train.batch_size=64', 'neural.train.amp=false', 'split.clusters_per_class=4'] if DEMO else []
PSEUDO_SHA = {}

def load_cfg(name='v3_base', extra=()):
    over = [f'lab_dir={LAB}'] + (['device=cpu'] + DEMO_SET if DEMO else []) + list(extra)
    c = config.load(REPO / 'configs' / f'{name}.json', over)
    if c['neural'].get('pseudo') is not None:
        c['neural']['pseudo']['sha'] = PSEUDO_SHA[c['neural']['pseudo']['file']]
    return c

cfg_of = lambda m: load_cfg(m.rsplit('_s', 1)[0], [f"neural.train.seed={m.rsplit('_s', 1)[1]}"])
def save(df, name):
    df.to_csv(Path(LAB) / f'v5_{name}.csv'); return df

R.setup(FIG)
cfg = load_cfg()
print('repo', REPO, '| lab', LAB, '| figures', FIG, '| device', cfg['device'])
print('code', next(l for l in (REPO / 'CHANGELOG.md').read_text().splitlines() if l.startswith('## ')))
""")
md("""
## 1. Préparation

Même cache et même split que la V3/V4 (validation par grappes de régimes, stress = carnets les moins profonds, audit scellé).
""")
code("""
files = None
if DEMO:
    from cfm.synthetic import create
    files = create('demo_data5')
sp = prepare(cfg, files)
y_all = np.asarray(io.load(LAB, 'train')['y']).astype(int)
ids = np.asarray(io.load(LAB, 'test')['ids']); classes = np.asarray(io.meta(LAB)['classes'])
print(json.loads((Path(LAB) / 'split.json').read_text())['sizes'])
""")
md("""
### Le parcours jusqu'ici

Chaque soumission a servi à trancher une question précise (registre complet : `LEADERBOARD.md`).
""")
code("""
LB_HISTORY = [('V2', .5072, '', False), ('R1', .5700, 'ticks/lots\\n+ SignatureNet', False),
              ('R1 équil.', .5958, 'Sinkhorn', False), ('R2', .59999, 'profondeur\\naugmentée', False),
              ('V3 A', .6028, '60 ép., EMA,\\nalignement', False), ('V3 B', .6351, 'vote entre\\nvoisins', False),
              ('V4 C', .6551, 'pseudo-labels\\n(tour 1)', False), ('V4 D', .6174, 'V4 C', True),
              ('V5 ALL', .6835, 'pseudo-labels\\n(tour 2)', False)]
R.lb_journey(LB_HISTORY); R.self_training_diagram();
""")
md("""
## 2. Le professeur V4 C et les pseudo-labels

On garde une fenêtre test si deux lectures du professeur concordent : la prédiction **lissée** (V4 C : voisins + équilibrage)
et la prédiction **brute** (moyenne des 5 modèles V4, équilibrée). Ensuite, pour chaque titre prédit, on prend les plus
confiantes jusqu'au quota q·N/24. Le quota est identique pour tous les titres : le professeur ne peut pas imposer ses
titres favoris. Comme les fenêtres sont classées par confiance, la sélection à 40 % est incluse dans celle à 60 %.
""")
code("""
if DEMO:   # pas de professeur réel : professeur bruité, pour vérifier la mécanique
    rng = np.random.default_rng(0); raw_p = rng.dirichlet(np.ones(len(classes)), len(ids)); C_p = (raw_p + rng.dirichlet(np.ones(len(classes)), len(ids))) / 2
else:
    tdir = Path(TEACHER_DIR) if TEACHER_DIR else REPO / 'teacher'
    Cz, Rz = np.load(tdir / 'v4_C_k40_probs.npz'), np.load(tdir / 'v4_raw_probs.npz')
    assert np.array_equal(Cz['obs_ids'], ids) and np.array_equal(Rz['obs_ids'], ids), 'professeur non aligné sur le test'
    C_p, raw_p = Cz['p'], Rz['p']
y_teacher, conf_teacher = C_p.argmax(1), C_p.max(1)
agree = y_teacher == raw_p.argmax(1)
SEL, rows = {}, []
for tag, q in QUOTAS.items():
    sel, ylab, tab = T.select_pseudo(C_p, raw_p, q)
    path = Path(LAB) / f'pseudo_v5_{tag}.npz'
    np.savez(path, obs_ids=ids, idx=sel, y=ylab)
    PSEUDO_SHA[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    SEL[tag] = sel; save(tab, f'pseudo_per_class_{tag}')
    rows.append({'quota': tag, 'selected': len(sel), 'share_of_test': len(sel) / len(ids),
                 'classes_below_quota': int((tab.selected < int(q * len(ids) / len(classes))).sum()),
                 'min_conf': float(conf_teacher[sel].min()), 'mean_conf': float(conf_teacher[sel].mean()),
                 'sha': PSEUDO_SHA[path.name]})
assert set(SEL['q40']) <= set(SEL['q60'])
pseudo_summary = save(pd.DataFrame(rows).set_index('quota'), 'pseudo_summary')
print(f'accord brut/lissé : {agree.mean():.3f}'); pseudo_summary.round(4)
""")
code("""
R.pseudo_selection(conf_teacher, agree, {'q60': SEL['q60'], 'q40': SEL['q40']}, y_teacher, classes);
""")
md("""
## 3. Les élèves

6 entraînements : `v5_q60` et `v5_q40`, 3 graines chacun. Un run terminé avec la même configuration (pseudo-labels
compris, vérifiés par empreinte) est réutilisé. Si les runs V3/V4 sont présents dans le lab, ils sont ajoutés aux figures
pour comparaison.
""")
code("""
def bal(run, part):
    z = np.load(Path(LAB) / 'runs' / run / f'{part}.npz')
    return scores(z['y'], blend.sinkhorn_balance(z['p']))['acc']

E = [f'v5_q60_s{s}' for s in SEEDS]; F = [f'v5_q40_s{s}' for s in SEEDS]; ALL = E + F
results = {m: fit(LAB, cfg_of(m), m, note='V5 : 2e tour d\\'auto-apprentissage') for m in ALL}
OLD = [r for r in ['v3_long_s0', 'v3_long_s1', 'v4_student_s0', 'v4_student_s1']
       if (Path(LAB) / 'runs' / r / 'stress.npz').exists()]
res = pd.DataFrame(results.values()).set_index('name')
for r in OLD:
    res.loc[r, ['best_epoch', 'minutes']] = [json.loads((Path(LAB) / 'runs' / r / 'result.json').read_text()).get(k) for k in ('best_epoch', 'minutes')]
res['config'] = [r.rsplit('_s', 1)[0] for r in res.index]
res['valid_bal'] = [bal(r, 'valid') for r in res.index]
res['stress_bal'] = [bal(r, 'stress') for r in res.index]
res['test_agree_teacher'] = [(np.load(Path(LAB) / 'runs' / r / 'test_dev.npz')['p'].argmax(1) == y_teacher).mean() for r in res.index]
save(res[['config', 'best_epoch', 'minutes', 'valid_bal', 'stress_bal', 'test_agree_teacher']], 'runs')
save(res.groupby('config')[['valid_bal', 'stress_bal', 'test_agree_teacher', 'minutes']].agg(['mean', 'std']), 'summary').round(4)
""")
code("""
groups = {'v3_long (sans pseudo)': ([r for r in OLD if r.startswith('v3')], R.C['v3']),
          'V4 élèves (tour 1, q40)': ([r for r in OLD if r.startswith('v4')], R.C['v4']),
          'V5 q40 (tour 2)': (F, R.C['q40']), 'V5 q60 (tour 2)': (E, R.C['q60'])}
R.learning_curves(LAB, groups); R.runs_scatter(res);
""")
md("## 4. Refit des 6 élèves (ensemble fixé d'avance)")
code("""
for m in ALL:
    refit(LAB, cfg_of(m), m, 'epochs')
table, w, agree_mat, oracle = blend.report(LAB, ALL); save(table, 'blend'); table.round(4)
""")
md("""
## 5. Alignement de profondeur

Les carnets du test sont moins profonds que ceux du train. On multiplie les tailles du test par γ = exp(m·Δ), où Δ est
l'écart de log-profondeur médiane. m est choisi sur le stress (qui imite ce décalage), et adopté seulement s'il apporte +0,3 pt.
""")
code("""
lv, names, _ = window_matrix(['levels'], LAB, 'train', cfg['lot']); j = names.index('levels:log_depth_q50')
lvt, _, _ = window_matrix(['levels'], LAB, 'test', cfg['lot'])
med = lambda x: float(np.nanmedian(x))
d_stress = med(lv[sp['fit'], j]) - med(lv[sp['stress'], j]); d_test = med(lv[:, j]) - med(lvt[:, j])
ys = y_all[sp['stress']]
Ps = np.mean([predict_shifted(LAB, cfg_of(m), m, 'dev', 'stress', [g * d_stress for g in GAMMA_MULTS]) for m in ALL], 0)
align = save(pd.DataFrame({'mult': GAMMA_MULTS, 'stress_bal': [scores(ys, blend.sinkhorn_balance(P))['acc'] for P in Ps]}), 'depth_alignment')
best_m = float(align.loc[align.stress_bal.idxmax(), 'mult'])
USE_ALIGN = best_m != 0 and align.stress_bal.max() >= align.loc[align.mult == 0, 'stress_bal'].iloc[0] + MIN_ALIGN_GAIN
g_test = best_m * d_test if USE_ALIGN else 0.0
print(f"Δ test {d_test:.3f} → γ test {np.exp(g_test):.3f} ({'adopté' if USE_ALIGN else 'rejeté'})"); align.round(4)
""")
code("""
Pt = {m: predict_shifted(LAB, cfg_of(m), m, 'refit', 'test', [g_test])[0] for m in ALL}
P_test = {'ALL': np.mean([Pt[m] for m in ALL], 0), 'E': np.mean([Pt[m] for m in E], 0), 'F': np.mean([Pt[m] for m in F], 0)}
""")
md("""
## 6. Vote entre voisins

La représentation `combo` concatène des statistiques de régime de la fenêtre et les représentations internes des élèves.
La grille est réglée sur le pool valid ∪ stress, qui contient ~5 fenêtres par titre-jour. Au test, il y en a ~20, d'où
k_test = 40 (≈ 4 × le meilleur k du pool, comme en V3/V4).
""")
code("""
pool = np.r_[sp['valid'], sp['stress']]; y_pool = y_all[pool]
member_pool = {m: np.r_[np.load(Path(LAB) / 'runs' / m / 'valid.npz')['p'], np.load(Path(LAB) / 'runs' / m / 'stress.npz')['p']] for m in ALL}
P_pool = np.mean(list(member_pool.values()), 0)
for m in ALL:
    embed(LAB, cfg_of(m), m, 'dev', ('valid', 'stress')); embed(LAB, cfg_of(m), m, 'refit', ('test',))
def combo(members, weights):
    if weights == 'dev':
        Zv, _ = T.emb_rep(LAB, members, 'dev', 'valid'); Zs, _ = T.emb_rep(LAB, members, 'dev', 'stress')
        return np.c_[T.window_rep(LAB, cfg, 'train', pool), T.standardise(np.r_[Zv, Zs])]
    Zt, eid = T.emb_rep(LAB, members, 'refit', 'test'); assert np.array_equal(eid, ids)
    return np.c_[T.window_rep(LAB, cfg, 'test', np.arange(len(ids))), T.standardise(Zt)]
g = save(T.grid(P_pool, y_pool, {'combo': combo(ALL, 'dev')}), 'neighbour_grid')
best = g[g.rep == 'combo'].iloc[0]; none = g[g.rep == '(none)'].iloc[0]
print(f"pool : sans {none.acc_balanced:.4f} | combo k={best.k} α={best.alpha} it={best.iters} → {best.acc_balanced:.4f}")
R.neighbour_heatmap(g); g.head(6).round(4)
""")
md("""
### Ce que chaque étape ajoute (pool valid ∪ stress)

Mesure interne, contaminée pour les élèves : elle sert à lire la *forme* de la cascade, pas son niveau.
""")
code("""
acc = lambda P: float((P.argmax(1) == y_pool).mean())
best_single = max(ALL, key=lambda m: acc(member_pool[m]))
steps = [(f'meilleur seul\\n({best_single})', acc(member_pool[best_single])), ('moyenne\\n6 élèves', acc(P_pool)),
         ('+ Sinkhorn', acc(blend.sinkhorn_balance(P_pool))), (f'+ voisins\\n(k={int(best.k)})', float(best.acc_balanced))]
R.cascade(steps, ylabel='Précision (pool valid ∪ stress)')
P_stress = {'q40 ×3': np.mean([np.load(Path(LAB) / 'runs' / m / 'stress.npz')['p'] for m in F], 0),
            'q60 ×3': np.mean([np.load(Path(LAB) / 'runs' / m / 'stress.npz')['p'] for m in E], 0)}
P_stress['6 élèves'] = (P_stress['q40 ×3'] + P_stress['q60 ×3']) / 2
if any(r.startswith('v4') for r in OLD):
    P_stress = {'V4 élèves': np.mean([np.load(Path(LAB) / 'runs' / r / 'stress.npz')['p'] for r in OLD if r.startswith('v4')], 0), **P_stress}
R.calibration(ys, P_stress)
R.recall_by_class(ys, {k: blend.sinkhorn_balance(P).argmax(1) for k, P in P_stress.items()}, classes)
R.confusion(ys, blend.sinkhorn_balance(P_stress['6 élèves']).argmax(1), classes);
""")
md("## 7. Soumissions")
code("""
def write(P, name, note):
    path = Path(LAB) / f'{name}.csv'
    pd.DataFrame({'obs_id': ids, 'eqt_code_cat': classes[P.argmax(1)]}).to_csv(path, index=False)
    np.savez(Path(LAB) / f'{name}_probs.npz', obs_ids=ids, p=P)
    print(f'{name:22s} {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}  ← {note}'); return P

out = {}
for tag, members, note in [('ALL', ALL, '6 élèves (q60 ×3 + q40 ×3)'), ('E', E, 'q60 ×3'), ('F', F, 'q40 ×3')]:
    Z = combo(members, 'refit')
    out[tag] = write(T.sinkhorn_balance(T.smooth(P_test[tag], T.knn(Z, K_TEST), float(best.alpha), int(best.iters))),
                     f'submission_v5_{tag}_k{K_TEST}', f'{note}, voisins k={K_TEST}')
mask60 = np.zeros(len(ids), bool); mask60[SEL['q60']] = True
diff = pd.DataFrame({k: {'vs_teacher_V4C': float((v.argmax(1) != y_teacher).mean()),
                         'vs_teacher_on_pseudo': float((v.argmax(1)[mask60] != y_teacher[mask60]).mean()),
                         'vs_teacher_off_pseudo': float((v.argmax(1)[~mask60] != y_teacher[~mask60]).mean()),
                         'vs_ALL': float((v.argmax(1) != out['ALL'].argmax(1)).mean())} for k, v in out.items()}).T
save(diff, 'submission_diffs').round(4)
""")
code("""
R.test_shift(C_p, {'V5 ALL': out['ALL'], 'V5 E (q60)': out['E'], 'V5 F (q40)': out['F']}, mask60)
Zt, _ = T.emb_rep(LAB, ALL, 'refit', 'test')
R.tsne_map(Zt, out['ALL'].argmax(1), classes, out['ALL'].max(1), n=1500 if DEMO else 4000);
""")
md("""
## 8. Décisions et archive

Ordre de soumission : **V5_ALL_k40**, puis E et F. Une fois les scores connus, les ajouter à `LB_HISTORY` et relancer la
cellule « parcours » pour mettre à jour la figure du rapport.
""")
code("""
decisions = {'teacher': 'V4 C_k40 (LB 0.6551)', 'agreement_raw_smoothed': float(agree.mean()),
             'pseudo': pseudo_summary.reset_index().to_dict('records'), 'members': {'ALL': ALL, 'E': E, 'F': F},
             'align': bool(USE_ALIGN), 'gamma_test': float(np.exp(g_test)),
             'neighbours': {'alpha': float(best.alpha), 'iters': int(best.iters), 'k_pool': int(best.k), 'k_test': K_TEST},
             'submit_order': [f'submission_v5_ALL_k{K_TEST}', f'submission_v5_E_k{K_TEST}', f'submission_v5_F_k{K_TEST}'],
             'reading': {'H1 (meilleur professeur)': 'ALL vs 0.6551', 'H2 (quota)': 'E vs F'}}
(Path(LAB) / 'v5_decisions.json').write_text(json.dumps(decisions, indent=2))
print((FIG / 'index.md').read_text())
""")
code("""
import zipfile
dest = Path(LAB).parent / ('demo_results_v5.zip' if DEMO else 'lab_results_v5.zip')
with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in Path(LAB).rglob('*'):
        rel = p.relative_to(LAB)
        if p.is_file() and not {'raw', 'features'} & set(rel.parts) and p.suffix != '.pt' and not p.name.startswith(('emb_', 'shift_')) \\
                and (rel.parts[0] != 'runs' or rel.parts[1].startswith('v5_')):
            z.write(p, rel)
    for p in FIG.glob('*'):
        z.write(p, Path('figures_v5') / p.name)
print(dest, round(dest.stat().st_size / 1e6, 1), 'Mo')
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
out = Path(__file__).resolve().parents[1] / 'CFM_V5.ipynb'
nbf.write(nb, out)
print(out)
