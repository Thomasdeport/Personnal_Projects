"""Generate the v_demo/ notebooks (python scripts/make_vdemo.py). Notebooks are git-ignored; this script is the source."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / 'v_demo'


def header(fig_dir, extra_imports='', use_torch=True):
    imports = ('import torch\nfrom cfm.demo import core, models, viz, train as T' if use_torch else
               'from cfm.demo import core, viz          # pas de torch ici (arbres / exploration)')
    lab_line = 'lab = core.open_lab(DEMO)' if use_torch else 'lab = core.open_lab(DEMO, torch_device=False)'
    return f"""
import os, sys, json, time
from pathlib import Path
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
CODE_DATASET = None   # ex. 'ton-username/cfm-signature-lab' ; None : code local ou Dataset attaché

def find_repo():
    if CODE_DATASET:
        import kagglehub
        path = Path(kagglehub.dataset_download(CODE_DATASET, force_download=True))
        return next(p.parent.parent for p in path.glob('**/cfm/__init__.py'))
    for p in [Path('.'), Path('..'), Path('../..'), Path('/kaggle/working/cfm-signature-lab')]:
        if (p / 'cfm' / '__init__.py').exists():
            return p.resolve()
    for p in Path('/kaggle/input').glob('**/cfm/__init__.py'):
        return p.parent.parent
    raise FileNotFoundError('Code introuvable : renseigner CODE_DATASET ou attacher le Dataset du code')

REPO = find_repo(); sys.path.insert(0, str(REPO))
DEMO = False          # True : données synthétiques sur CPU (vérifie que tout tourne ; scores sans valeur)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
{imports}
{extra_imports}
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
viz.setup('figures/{fig_dir}')
{lab_line}
LABELS = [str(c) for c in lab.classes]
""".strip()


FOOTER = """
import shutil
archive = shutil.make_archive(str(viz.FIG_DIR), 'zip', viz.FIG_DIR)
print('Figures :', sorted(p.name for p in viz.FIG_DIR.glob('*.png')), '→', archive)
""".strip()


def notebook(name, cells):
    nb = nbf.v4.new_notebook(metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src.strip()) if kind == 'md' else nbf.v4.new_code_cell(src.strip()))
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return path


# ------------------------------------------------------------------ V1
V1 = [
('md', """
# V1 — Benchmark : que vaut-on sans rien d'intelligent ?

**Question.** Avant tout réseau, quels scores obtient-on avec des modèles classiques sur les **statistiques de fenêtre
de la première version** (moyennes et écarts-types de canaux relatifs + fréquences des catégories) ?

**Protocole commun à tout `v_demo`.**
- `fit` entraîne.
- `valid` (grappes de régimes tenues à l'écart) choisit.
- `stress` (carnets peu profonds, comme le test) diagnostique.
- Le hasard vaut 1/24 ≈ 0,042.

Durée : environ 5 à 10 min (CPU suffit).
"""),
('code', header('V1_benchmark', use_torch=False, extra_imports='from sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.preprocessing import StandardScaler')),
('code', """
X, names = lab.table(core.OLD_STATS)
fit, valid, stress = lab.split['fit'], lab.split['valid'], lab.split['stress']
med = np.nanmedian(X[fit], 0); X = np.where(np.isfinite(X), X, med)
sc = StandardScaler().fit(X[fit]); Xs = sc.transform(X)
print(f'{X.shape[1]} statistiques par fenêtre :', names[:6], '…')
"""),
('code', """
rows, probs = [], {}
def run(nom, make, Xm=Xs, sub=None):
    t0 = time.time(); tr = fit if sub is None else np.random.default_rng(0).choice(fit, min(sub, len(fit)), replace=False)
    m = make().fit(Xm[tr], lab.y[tr])
    out = {'modèle': nom}
    for part, idx in [('valid', valid), ('stress', stress)]:
        p = m.predict_proba(Xm[idx]); probs[(nom, part)] = p
        out[part] = core.accuracy(lab.y[idx], p); out[part + '_équilibré'] = core.accuracy(lab.y[idx], p, True)
    out['minutes'] = round((time.time() - t0) / 60, 2); rows.append(out); print(out, flush=True)

class Majority:
    def fit(self, X, y): self.p = np.bincount(y, minlength=len(LABELS)) / len(y); return self
    def predict_proba(self, X): return np.tile(self.p, (len(X), 1))

run('Classe majoritaire', Majority)
run('Naive Bayes gaussien', GaussianNB)
run('Régression logistique', lambda: LogisticRegression(max_iter=300, C=1.0))
run('k plus proches voisins (k=25)', lambda: KNeighborsClassifier(25, n_jobs=-1), sub=40000)
run('Forêt aléatoire (200 arbres)', lambda: RandomForestClassifier(200, max_depth=20, min_samples_leaf=3, n_jobs=-1, random_state=0), Xm=X)
res = pd.DataFrame(rows); res
"""),
('code', """
viz.compare(res, ('valid', 'stress'), 'V1 — modèles classiques sur les statistiques de la première version', 'v1_compare')
best = res.sort_values('valid', ascending=False).iloc[0]['modèle']
viz.confusion(lab.y[valid], probs[(best, 'valid')], LABELS, f'Confusion — {best} (valid)', 'v1_confusion')
viz.recall_by_class(lab.y[valid], {best: probs[(best, 'valid')]}, LABELS, f'Rappel par titre — {best}', 'v1_recall')
"""),
('md', """
**À retenir.**
- Ces scores sont le **plancher**.
- Des statistiques relatives, même avec un bon modèle d'arbres, plafonnent bas.
- L'écart entre `valid` et `stress` montre déjà le problème du changement de régime.

La suite (V2) teste si de **meilleures features** relèvent ce plafond, avant d'aller vers les séquences (V3).
"""),
('code', FOOTER),
]

# ------------------------------------------------------------------ V2
V2 = [
('md', """
# V2 — Modèles d'arbres : l'effet des features

**Question.** Avec le même modèle (XGBoost), combien rapportent les features du papier (ticks, lots, position,
parcours d'ordres…) par rapport aux statistiques relatives de la première version ? Puis LightGBM et CatBoost
sur l'ensemble complet, l'importance des features, et la carte utilité/dérive.

Durée : environ 10 à 20 min sur GPU.
"""),
('code', header('V2_tree_models', 'from cfm import tabular, screen, plots as cplots\nfrom cfm.registry import BLOCKS', use_torch=False)),
('code', """
ALL = ['base_relative', 'cat_freq', 'cat_cross', 'levels', 'ticks', 'lots', 'position', 'orders', 'book_dyn', 'tokens_bow', 'transitions']
fit, valid, stress = lab.split['fit'], lab.split['valid'], lab.split['stress']
tr, es = tabular.inner_split(fit, lab.y, 0)          # arrêt anticipé sur une tranche interne du fit
DEV = 'cuda' if core.has_cuda() else 'cpu'
rows, probs, importances = [], {}, {}

def run(nom, blocks, model='xgb', params=None):
    X, names = lab.table(blocks); t0 = time.time()
    if model == 'catboost':
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=1500, learning_rate=.1, depth=6, loss_function='MultiClass', verbose=0,
                               task_type='GPU' if DEV == 'cuda' else 'CPU', random_seed=0, early_stopping_rounds=50)
        m.fit(X[tr], lab.y[tr], eval_set=(X[es], lab.y[es])); pred = m.predict_proba
    else:
        pred, _ = tabular.train(X, lab.y, tr, es, model, params, 0, DEV if model == 'xgb' else 'cpu')
        if model == 'xgb':
            m = pred.__self__; importances[nom] = pd.Series(m.feature_importances_, index=names)
    out = {'modèle': nom, 'features': X.shape[1]}
    for part, idx in [('valid', valid), ('stress', stress)]:
        p = pred(X[idx]); probs[(nom, part)] = p
        out[part] = core.accuracy(lab.y[idx], p); out[part + '_équilibré'] = core.accuracy(lab.y[idx], p, True)
    out['minutes'] = round((time.time() - t0) / 60, 1); rows.append(out); print(out, flush=True)
"""),
('code', """
run('XGBoost — stats première version', core.OLD_STATS)
run('XGBoost — + ticks', core.OLD_STATS + ['ticks'])
run('XGBoost — + ticks + lots', core.OLD_STATS + ['ticks', 'lots'])
run('XGBoost — + ticks + lots + position + ordres', core.OLD_STATS + ['ticks', 'lots', 'position', 'orders'])
run('XGBoost — toutes les features', ALL)
run('LightGBM — toutes les features', ALL, 'lgbm')
run('CatBoost — toutes les features', ALL, 'catboost')
res = pd.DataFrame(rows); res
"""),
('code', """
viz.compare(res, ('valid', 'stress'), 'V2 — arbres : chaque famille de features ajoutée une à une', 'v2_compare')
imp = importances['XGBoost — toutes les features'].sort_values(ascending=False).head(25)
fam = [BLOCKS[n.split(':')[0]].family for n in imp.index]
cols = {f: viz.CYCLE[i % len(viz.CYCLE)] for i, f in enumerate(sorted(set(fam)))}
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(range(len(imp)), imp.values, color=[cols[f] for f in fam]); ax.set_yticks(range(len(imp)), imp.index, fontsize=7); ax.invert_yaxis()
for f, c in cols.items(): ax.barh([], [], color=c, label=f)
ax.legend(fontsize=7); ax.set_title('25 features les plus utilisées par XGBoost (gain)'); viz.save(fig, 'v2_importance')
"""),
('code', """
# Utilité (η², sur le fit) contre dérive (KS train/test) : la carte qui a guidé tout le projet
uni = screen.univariate(lab.dir, lab.cfg, ALL, base=core.OLD_STATS)
cplots.screen_map(uni, lab.dir); fig = plt.gcf(); viz.save(fig, 'v2_utility_vs_drift')
uni.head(15)[['feature', 'family', 'eta2', 'ks', 'shift_sd']].round(3)
"""),
('code', """
# Où se trouve le décalage train/test ? AUC d'un classifieur train-vs-test, bloc par bloc
adv = screen.adversarial(lab.dir, lab.cfg, ALL, n=20000, device=DEV)
cplots.adversarial_bars(adv, lab.dir); viz.save(plt.gcf(), 'v2_adversarial'); adv.round(3)
"""),
('md', """
**À retenir.**
- Ajouter les ticks et les lots fait monter **le même XGBoost** de plusieurs points : c'est la représentation qui compte.
- Les arbres plafonnent pourtant loin des réseaux : ils voient des **fréquences**, pas l'**ordre** des événements (V3).
- La carte utilité/dérive montre les colonnes de tick en haut à gauche : séparatrices **et** stables.
"""),
('code', FOOTER),
]

# ------------------------------------------------------------------ V3
V3 = [
('md', """
# V3 — Modèles séquentiels sur l'ancienne représentation

**Question.** Lire les 100 événements dans l'ordre apporte-t-il quelque chose par rapport aux seules statistiques ?
Tous les modèles ont ici la **même entrée que la première version** : canaux relatifs à des médianes locales
+ 4 catégories (venue, action, côté, trade). Aucun tick, aucun lot.

Durée : 4 modèles × 15 époques, environ 15 min sur GPU.
"""),
('code', header('V3_sequence_models')),
('code', """
data = lab.sequences('old')
print('tokens :', core.OLD_TOKENS, '| canaux :', data['cont_names'], '| contexte :', data['ctx'].shape[1], 'colonnes')
EPOCHS = 3 if DEMO else 15
zoo = {
    'MLP sur statistiques': lambda: models.StatsMLP(data['ctx'].shape[1]),
    'CNN (3 conv. dilatées)': lambda: models.TokenCNN(data['cards'], data['cont'].shape[-1], data['ctx'].shape[1]),
    'GRU bidirectionnel': lambda: models.TinyGRU(data['cards'], data['cont'].shape[-1], data['ctx'].shape[1]),
    'Petit Transformer': lambda: models.TinyTransformer(data['cards'], data['cont'].shape[-1], data['ctx'].shape[1]),
}
rows, hists, probs = [], {}, {}
for nom, make in zoo.items():
    print('▶', nom); t0 = time.time()
    m, h, norm, dev = T.train(make(), lab, data, epochs=EPOCHS)
    out, p = T.evaluate(m, lab, data, norm, dev, nom, time.time() - t0)
    rows.append(out); hists[nom] = h; probs[nom] = p
res = pd.DataFrame(rows); res.round(4)
"""),
('code', """
viz.compare(res, ('valid', 'stress'), 'V3 — lire la séquence (ancienne représentation)', 'v3_compare')
viz.curves(hists, 'V3 — apprentissage', 'v3_curves')
viz.recall_by_class(lab.y[lab.split['valid']], {k: v['valid'] for k, v in probs.items()}, LABELS, 'Rappel par titre (valid)', 'v3_recall')
"""),
('md', """
**À retenir.**
- Passer des statistiques (MLP) à la lecture de la séquence (CNN, GRU, Transformer) fait gagner beaucoup.
- L'information est dans **l'enchaînement des événements**.
- Mais tous ces modèles voient encore des grandeurs relatives à la fenêtre. V4 reconstruit le meilleur modèle de cette famille (0,507). V5 montre ce que change la **représentation**.
"""),
('code', FOOTER),
]

# ------------------------------------------------------------------ V4
V4 = [
('md', """
# V4 — La meilleure version d'avant (0,507), reconstruite simplement

**Ce qu'était la V2 (record public 0,5072).**
- Un Transformer hybride : canaux relatifs + 4 embeddings de catégories, 3 blocs d'attention, pooling par attention.
- Une branche de statistiques, fusionnée avec la séquence.
- Un ensemble de deux variantes × deux graines.

⚠️ **C'est une reconstruction simplifiée**, pas le run exact :
- pas de biais « même ordre » ;
- pas de loss contrastive ;
- statistiques approchées ;
- budget court.

Elle sert de **point de comparaison honnête** pour V5, avec le même protocole et les mêmes données.

Durée : 2 graines × environ 8 min sur GPU.
"""),
('code', header('V4_best_version_0507')),
('code', """
data = dict(lab.sequences('old'))
data['ctx'] = lab.table(core.OLD_STATS + ['levels'])[0]       # la V2 gardait aussi des niveaux dans sa branche statistique
EPOCHS = 3 if DEMO else 20
rows, hists, P = [], {}, {'valid': [], 'stress': []}
mods = []
for seed in [0, 1]:
    print('▶ graine', seed); t0 = time.time()
    m, h, norm, dev = T.train(models.V2Hybrid(data['cards'], data['cont'].shape[-1], data['ctx'].shape[1]), lab, data,
                              epochs=EPOCHS, lr=1e-3, seed=seed)
    out, p = T.evaluate(m, lab, data, norm, dev, f'Hybride V2 (graine {seed})', time.time() - t0)
    rows.append(out); hists[f'graine {seed}'] = h; mods.append((m, norm))
    for k in P: P[k].append(p[k])
ens = {'modèle': 'Ensemble des 2 graines (comme la V2)'}
for part in ['valid', 'stress']:
    p = np.mean(P[part], 0); y = lab.y[lab.split[part]]
    ens[part] = T.accuracy(y, p); ens[part + '_équilibré'] = T.accuracy(y, p, True)
rows.append(ens); res = pd.DataFrame(rows); res.round(4)
"""),
('code', """
viz.curves(hists, 'V4 — hybride V2 reconstruit', 'v4_curves')
viz.compare(res, ('valid', 'stress'), 'V4 — la meilleure version d\\'avant (reconstruction)', 'v4_compare')
viz.confusion(lab.y[lab.split['stress']], np.mean(P['stress'], 0), LABELS, 'V4 — confusion sur le stress', 'v4_confusion')
"""),
('code', """
# Soumission de contrôle (même règle que la V2 : argmax, sans équilibrage)
data_t = dict(lab.sequences('old', 'test')); data_t['ctx'] = lab.table(core.OLD_STATS + ['levels'], 'test')[0]
dev_t = T.OnDevice(data_t, lab.device)
Pt = np.mean([T.predict(m, dev_t, n, np.arange(len(lab.test_ids))) for m, n in mods], 0)
pd.DataFrame({'obs_id': lab.test_ids, 'eqt_code_cat': lab.classes[Pt.argmax(1)]}).to_csv('submission_V4_hybride_v2.csv', index=False)
print('submission_V4_hybride_v2.csv écrit — score public de référence de la vraie V2 : 0,5072')
"""),
('md', """
**À retenir.**
- C'est le niveau de départ.
- Les scores de `valid` sont bien au-dessus du LB, parce que des fenêtres de jours voisins se ressemblent.
- `stress` est plus sévère.

V5 part d'un modèle **plus petit** que celui-ci et montre, étape par étape, ce qui fait gagner.
"""),
('code', FOOTER),
]

# ------------------------------------------------------------------ V5
V5 = [
('md', """
# V5 — Un modèle simple, les améliorations une par une

On part d'un **petit CNN** (3 convolutions, environ 100 k paramètres) qui lit l'**ancienne représentation**. Puis on ajoute
**une seule chose à chaque étape**, en mesurant sur les mêmes fenêtres (`valid ∪ stress`) :

| Étape | Ce qu'on ajoute | Pourquoi (en une phrase) |
|---|---|---|
| 0 | Ancienne représentation | Grandeurs divisées par des médianes de la fenêtre : un spread de 1 tick et un de 5 ticks se ressemblent |
| 1 | **Tokens en ticks et en lots** | Le tick et le lot sont des conventions de marché propres à chaque titre, et elles ne changent pas d'une période à l'autre |
| 2 | **Augmentation de profondeur** | Le test est moins profond (prix plus élevés ?) : on apprend qu'un titre peut afficher plus ou moins d'actions |
| 3 | **Entraîner plus longtemps** | Les courbes montaient encore |
| 4 | **Moyenne de 3 graines** | Réduit le bruit d'un entraînement |
| 5 | **Équilibrage (Sinkhorn)** | Le décalage pousse le modèle vers les titres peu profonds ; on remet ~1/24 du test sur chaque titre |
| 6 | **Vote entre voisins** | Le test contient 20 fenêtres par titre et par jour : des fenêtres semblables votent ensemble |

Les étapes 5 et 6 utilisent le test sans labels (transductif). Durée totale : environ 15 à 20 min sur GPU.
"""),
('code', header('V5_simple_improvements', 'from cfm.blend import sinkhorn_balance\nfrom cfm import transductive as TR')),
('code', """
pool = np.r_[lab.split['valid'], lab.split['stress']]; y_pool = lab.y[pool]
acc = lambda P: float((P.argmax(1) == y_pool).mean())
E = 3 if DEMO else 20
ladder, hists, keep = [], {}, {}

def step(nom, kind, epochs, sigma=0., seed=0):
    data = lab.sequences(kind); t0 = time.time()
    m = models.TokenCNN(data['cards'], data['cont'].shape[-1], data['ctx'].shape[1])
    print(f'▶ {nom} — {models.n_params(m):,} paramètres')
    m, h, norm, dev = T.train(m, lab, data, epochs=epochs, depth_sigma=sigma, seed=seed)
    P = T.predict(m, dev, norm, pool); hists[nom] = h
    print(f'   accuracy valid ∪ stress = {acc(P):.4f} ({(time.time() - t0) / 60:.1f} min)')
    return m, norm, dev, P
"""),
('code', """
_, _, _, P0 = step('0 · ancienne représentation', 'old', E);                ladder.append(('0 · ancienne\\nreprésentation', acc(P0)))
_, _, _, P1 = step('1 · + tokens ticks/lots', 'new', E);                     ladder.append(('1 · + tokens\\nticks / lots', acc(P1)))
_, _, _, P2 = step('2 · + augmentation profondeur', 'new', E, .3);          ladder.append(('2 · + profondeur', acc(P2)))
m3, n3, d3, P3 = step('3 · + entraînement 2× plus long', 'new', 2 * E, .3); ladder.append(('3 · + 2× plus\\nlong', acc(P3)))
"""),
('code', """
members = [(m3, n3, d3, P3)]
for seed in [1, 2]:
    members.append(step(f'4 · graine {seed}', 'new', 2 * E, .3, seed))
P4 = np.mean([x[3] for x in members], 0);       ladder.append(('4 · moyenne\\n3 graines', acc(P4)))
P5 = sinkhorn_balance(P4);                      ladder.append(('5 · + équilibrage', acc(P5)))
# 6 · voisins : représentation = statistiques de régime ⊕ vecteurs internes du modèle, voisins cherchés dans le pool
Z = np.mean([T.predict(m, d, n, pool, features=True) for m, n, d, _ in members], 0)
W = TR.window_rep(lab.dir, lab.cfg, 'train', pool)
Zc = np.c_[W, TR.standardise(Z)]
P6 = sinkhorn_balance(TR.smooth(P4, TR.knn(Zc, 10), .5, 5)); ladder.append(('6 · + vote entre\\nvoisins', acc(P6)))
pd.DataFrame(ladder, columns=['étape', 'accuracy valid ∪ stress']).round(4)
"""),
('code', """
viz.waterfall(ladder, 'V5 — chaque amélioration, une à la fois (même modèle simple, mêmes fenêtres)', 'v5_ladder', 'Accuracy valid ∪ stress')
viz.curves({k: v for k, v in hists.items() if not k.startswith('4')}, 'V5 — apprentissage des étapes 0 à 3', 'v5_curves')
viz.waterfall([('V2 (record\\nprécédent)', .5072), ('SignatureNet\\nblend 4', .5700), ('+ Sinkhorn', .5958), ('+ profondeur', .59999),
               ('+ 60 ép.,\\nalignement', .6028), ('+ voisins', .6351)],
              'Scores PUBLICS réels du pipeline complet (pour comparaison)', 'v5_ladder_public', 'Accuracy publique')
"""),
('code', """
y_v = lab.y[lab.split['valid']]; nv = len(y_v)
viz.confusion(y_pool, P4, LABELS, 'Avant : moyenne de 3 graines (sans post-traitement)', 'v5_confusion_before')
viz.confusion(y_pool, P6, LABELS, 'Après : équilibrage + vote entre voisins', 'v5_confusion_after')
viz.recall_by_class(y_pool, {'0 · ancienne': P0, '4 · 3 graines': P4, '6 · + voisins': P6}, LABELS, 'Rappel par titre', 'v5_recall')
Zt_data = lab.sequences('new', 'test'); dt = T.OnDevice(Zt_data, lab.device)
Zt = T.predict(m3, dt, n3, np.arange(len(lab.test_ids)), features=True)
viz.embedding_map(T.predict(m3, d3, n3, lab.split['valid'], features=True), y_v, LABELS,
                  'Représentations du modèle : fenêtres valid (couleur = titre), test en gris', 'v5_tsne', extra=(Zt, 'test'))
"""),
('code', """
# Soumissions pour vérifier soi-même : étapes 4, 5 et 6 sur le test (voisins : k ×4 car le test a ~20 fenêtres par titre-jour)
idx_t = np.arange(len(lab.test_ids))
Pt = np.mean([T.predict(m, dt, n, idx_t) for m, n, _, _ in members], 0)
Zt_all = np.mean([T.predict(m, dt, n, idx_t, features=True) for m, n, _, _ in members], 0)
Zc_t = np.c_[TR.window_rep(lab.dir, lab.cfg, 'test', idx_t), TR.standardise(Zt_all)]
for nom, P in [('etape4_moyenne', Pt), ('etape5_equilibre', sinkhorn_balance(Pt)),
               ('etape6_voisins', sinkhorn_balance(TR.smooth(Pt, TR.knn(Zc_t, 40), .5, 5)))]:
    pd.DataFrame({'obs_id': lab.test_ids, 'eqt_code_cat': lab.classes[P.argmax(1)]}).to_csv(f'submission_V5_{nom}.csv', index=False)
    print('écrit : submission_V5_' + nom + '.csv')
"""),
('md', """
**À retenir.**
1. Le gain le plus net vient de la **représentation** : les tokens en ticks et en lots.
2. Les deux post-traitements (**équilibrage**, **vote entre voisins**) ne demandent aucun réentraînement et pèsent lourd au LB : environ +2,6 et +3,2 points dans le pipeline complet.
3. Le modèle lui-même peut rester **petit**.

Sur `valid ∪ stress`, l'effet du vote entre voisins est **sous-estimé** : le pool contient environ 5 fenêtres par titre-jour, contre 20 dans le test.
"""),
('code', FOOTER),
]

# ------------------------------------------------------------------ data exploration
TEACHER = """
teacher = REPO / 'teacher' / 'v3_B_probs.npz'
if teacher.exists() and not DEMO:
    z = np.load(teacher); assert np.array_equal(z['obs_ids'], lab.test_ids)
    y_test_pred = z['p'].argmax(1); TEST_LABEL = 'test (titre prédit par V3 B, LB 0,635)'
else:
    y_test_pred, TEST_LABEL = None, None
print('titres du test :', TEST_LABEL or 'indisponibles (barres test masquées)')
def by_class(X, names, cols, y):
    df = pd.DataFrame(X, columns=names)[cols]; df['y'] = y; return df.groupby('y').mean()
"""

D1 = [
('md', """
# D1 — Anatomie d'une fenêtre

**Question.** À quoi ressemblent concrètement 100 événements ?
- Comment un ordre vit-il dans une fenêtre ?
- Qu'est-ce qui distingue, à l'œil, un titre « à gros tick » d'un titre « à petit tick » ?

Durée : < 2 min.
"""),
('code', header('D1_anatomy', use_torch=False) + '\nfrom cfm import io\nfrom cfm.features.common import Raw\nfrom cfm.registry import context'),
('code', """
raw = io.load(lab.dir, 'train')
X, names = lab.table(['ticks'])
s1 = X[:, names.index('ticks:spread_1')]
cls = pd.Series(s1).groupby(lab.y).mean()
big, small = int(cls.idxmax()), int(cls.idxmin())
print(f'titre au plus gros tick : {LABELS[big]} (spread à 1 tick {cls[big]:.0%} du temps) | au plus petit : {LABELS[small]} ({cls[small]:.0%})')
pick = {f'gros tick (titre {LABELS[big]})': np.flatnonzero(lab.y == big)[0], f'petit tick (titre {LABELS[small]})': np.flatnonzero(lab.y == small)[0]}
for k, i in pick.items():
    viz.window_anatomy({key: np.asarray(raw[key][i]) for key in ['num', 'cat', 'oid']}, io.meta(lab.dir)['tick'],
                       f'Une fenêtre réelle — {k}', 'd1_window_' + ('big' if 'gros' in k else 'small'))
"""),
('code', """
seq = lab.sequences('new')
i = pick[list(pick)[0]]
viz.token_strip(seq['tok'][i], [n.replace('tok_', '') for n in core.NEW_TOKENS], title='Les 40 premiers événements de la même fenêtre, en tokens', name='d1_tokens')
r = Raw({k: np.asarray(raw[k][i:i + 1]) for k in ['num', 'cat', 'oid']}, context(lab.dir, lab.cfg['lot']))
pd.DataFrame({'action': r.action[0], 'côté': r.side[0], 'trade': r.is_trade[0], 'spread (ticks)': r.spread_t[0],
              'Δmid (½ ticks)': r.dmid_h[0], 'distance (ticks)': r.own_dist[0], 'flux (lots)': r.flux_lots[0],
              'ordre': r.oid[0], 'occurrence': r.occ[0], 'action préc.': r.prev_action[0]}).head(20).round(2)
"""),
('code', """
# Combien d'événements vit un ordre dans une fenêtre, et comment apparaît-il la première fois ?
O, on = lab.table(['orders'])
df = pd.DataFrame(O, columns=on)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
axs[0].hist(df['orders:orders_per_event'] * 100, bins=40, color=viz.PAL['blue']); axs[0].set_title('Ordres distincts par fenêtre'); axs[0].set_xlabel('nombre d\\'ordres')
df[['orders:first_seen_A', 'orders:first_seen_U', 'orders:first_seen_D']].mean().rename(lambda s: s.split('_')[-1]).plot.bar(ax=axs[1], color=[viz.PAL['new'], viz.PAL['purple'], viz.PAL['old']])
axs[1].set_title('Première apparition d\\'un ordre (U ou D = ordre antérieur à la fenêtre)'); axs[1].set_ylabel('part des ordres')
viz.save(fig, 'd1_orders')
"""),
('code', FOOTER),
]

D2 = [
('md', """
# D2 — Les signatures des titres

**Question.** Qu'est-ce qui distingue un titre d'un autre, et est-ce que ça tient entre le train et le test ?

Le test n'a pas de labels. Pour le découper par titre, on utilise le titre **prédit** par la meilleure soumission (V3 B, 0,635). C'est approximatif, mais suffisant pour voir les tendances.

Durée : < 3 min.
"""),
('code', header('D2_stock_signatures', use_torch=False)),
('code', TEACHER),
('code', """
def shares(blocks, cols, title, name):
    Xtr, nm = lab.table(blocks); tr = by_class(Xtr, nm, cols, lab.y)
    te = by_class(lab.table(blocks, 'test')[0], nm, cols, y_test_pred) if y_test_pred is not None else tr * np.nan
    tr.columns = te.columns = [c.split(':')[1] for c in cols]
    return viz.per_class_shares(tr, te, list(tr.columns), LABELS, title, name, TEST_LABEL or 'test')

shares(['ticks'], ['ticks:spread_1', 'ticks:spread_2', 'ticks:spread_3_4', 'ticks:spread_5p'],
       'Régime de tick : part du temps à 1, 2, 3–4, ≥5 ticks de spread', 'd2_ticks')
shares(['lots'], ['lots:flux_round', 'lots:flux_odd', 'lots:flux_mixed'], 'Conventions de taille : lots ronds, odd lots, tailles mixtes', 'd2_lots')
shares(['cat_freq'], [f'cat_freq:venue={v}' for v in range(1, 7)], 'Répartition des événements entre venues', 'd2_venues')
shares(['position'], ['position:pos_best', 'position:pos_d1', 'position:pos_d2', 'position:pos_d3_5', 'position:pos_d6p'],
       'Où arrivent les événements (distance au meilleur prix)', 'd2_position')
"""),
('code', """
# Profondeur par titre : train contre test
Xtr, nm = lab.table(['levels']); j = nm.index('levels:log_depth_q50')
fig, ax = plt.subplots(figsize=(11, 3.4))
pos = np.arange(len(LABELS))
ax.boxplot([Xtr[lab.y == c, j] for c in range(len(LABELS))], positions=pos - .2, widths=.35, showfliers=False,
           patch_artist=True, boxprops=dict(facecolor=viz.PAL['blue'], alpha=.6))
if y_test_pred is not None:
    Xte = lab.table(['levels'], 'test')[0]
    ax.boxplot([Xte[y_test_pred == c, j] for c in range(len(LABELS))], positions=pos + .2, widths=.35, showfliers=False,
               patch_artist=True, boxprops=dict(facecolor=viz.PAL['orange'], alpha=.6))
ax.set_xticks(pos, LABELS); ax.set_ylabel('log(1 + profondeur médiane / lot)')
ax.set_title('Profondeur affichée par titre — bleu : train, orange : ' + (TEST_LABEL or 'test indisponible'))
viz.save(fig, 'd2_depth')
"""),
('code', """
# Empreinte « sac de mots » : fréquence des tokens action × côté × zone, titres regroupés par ressemblance
from scipy.cluster.hierarchy import leaves_list, linkage
Xb, nb_ = lab.table(['tokens_bow'])
M = by_class(Xb, nb_, nb_, lab.y); M = (M - M.mean()) / M.std()
o = leaves_list(linkage(M.values, 'average'))
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(M.values[o], aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_yticks(range(len(o)), np.asarray(LABELS)[o]); ax.set_xticks(range(M.shape[1]), [c.split(':')[1] for c in M.columns], rotation=90, fontsize=6)
ax.set_title('Empreinte de chaque titre (écarts-types par rapport à la moyenne des titres)'); ax.grid(False)
fig.colorbar(im, ax=ax, shrink=.7); viz.save(fig, 'd2_token_fingerprint')
"""),
('code', FOOTER),
]

D3 = [
('md', """
# D3 — Ce qui change entre le train et le test

**Question.** Le test vient d'une autre période : qu'est-ce qui a bougé, et est-ce compatible avec l'hypothèse de prix plus élevés (moins d'actions affichées, plus d'odd lots) ?

Durée : environ 5 min.
"""),
('code', header('D3_train_test_shift', 'from cfm import screen, plots as cplots', use_torch=False)),
('code', """
ALL = ['base_relative', 'cat_freq', 'levels', 'ticks', 'lots', 'position', 'orders', 'book_dyn']
uni = screen.univariate(lab.dir, lab.cfg, ALL, base=core.OLD_STATS)
cplots.screen_map(uni, lab.dir); viz.save(plt.gcf(), 'd3_utility_vs_drift')
uni.sort_values('ks', ascending=False).head(12)[['feature', 'eta2', 'ks', 'shift_sd']].round(3)
"""),
('code', """
top = uni.sort_values('ks', ascending=False).head(6).feature.tolist()
fig, axs = plt.subplots(2, 3, figsize=(11, 5.4))
for a, f in zip(axs.ravel(), top):
    blk, col = f.split(':'); Xtr, nm = lab.table([blk]); Xte, _ = lab.table([blk], 'test'); j = nm.index(f)
    lo, hi = np.nanquantile(np.r_[Xtr[:, j], Xte[:, j]], [.005, .995]); bins = np.linspace(lo, hi, 40)
    a.hist(Xtr[:, j], bins, density=True, histtype='step', color=viz.PAL['blue'], label='train')
    a.hist(Xte[:, j], bins, density=True, histtype='step', color=viz.PAL['orange'], label='test'); a.set_title(f, fontsize=8)
axs[0, 0].legend(fontsize=7); fig.suptitle('Les 6 colonnes les plus décalées (KS)', fontweight='bold'); viz.save(fig, 'd3_top_shifts')
"""),
('code', """
# Hypothèse « prix plus élevés » : moins de profondeur ET plus d'odd lots doivent aller ensemble
L, ln = lab.table(['levels', 'lots']); Lt, _ = lab.table(['levels', 'lots'], 'test')
d, o = ln.index('levels:log_depth_q50'), ln.index('lots:flux_odd')
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True, sharey=True)
for a, M_, t in [(axs[0], L, 'train'), (axs[1], Lt, 'test')]:
    a.hexbin(M_[:, d], M_[:, o], gridsize=40, cmap='Blues', mincnt=1, bins='log'); a.set_title(t)
    a.set_xlabel('log(1 + profondeur médiane / lot)')
axs[0].set_ylabel('part d\\'odd lots dans le flux')
fig.suptitle('Profondeur contre odd lots', fontweight='bold'); viz.save(fig, 'd3_depth_vs_oddlots')
print('médianes — profondeur : train %.3f / test %.3f | odd lots : train %.3f / test %.3f' %
      (np.nanmedian(L[:, d]), np.nanmedian(Lt[:, d]), np.nanmean(L[:, o]), np.nanmean(Lt[:, o])))
"""),
('code', """
adv = screen.adversarial(lab.dir, lab.cfg, ALL, n=20000, device='cuda' if core.has_cuda() else 'cpu')
cplots.adversarial_bars(adv, lab.dir); viz.save(plt.gcf(), 'd3_adversarial'); adv.round(3)
"""),
('code', FOOTER),
]

D4 = [
('md', """
# D4 — Comment vivent les ordres

**Question.** Les titres se distinguent-ils par la façon dont leurs ordres sont ajoutés, modifiés, annulés ou exécutés ?

Tout est mesuré **dans** des fenêtres tronquées de 100 événements : ce ne sont pas des durées de vie complètes.

Durée : < 2 min.
"""),
('code', header('D4_order_lifecycles', use_torch=False)),
('code', TEACHER),
('code', """
O, on = lab.table(['orders'])
trans = [c for c in on if '_to_' in c]
M = by_class(O, on, trans, lab.y)
fig, axs = plt.subplots(4, 6, figsize=(12, 8))
for c, a in zip(range(len(LABELS)), axs.ravel()):
    a.imshow(M.loc[c].values.reshape(3, 3), cmap='Purples', vmin=0, vmax=M.values.max())
    a.set_xticks(range(3), list('ADU'), fontsize=6); a.set_yticks(range(3), list('ADU'), fontsize=6); a.set_title(f'titre {LABELS[c]}', fontsize=7); a.grid(False)
fig.suptitle('Transitions d\\'un même ordre (ligne : action précédente → colonne : action suivante)', fontweight='bold')
viz.save(fig, 'd4_transitions_by_stock')
"""),
('code', """
cols = ['orders:repeat_share', 'orders:multi_event_orders', 'orders:added_then_deleted', 'orders:D_trade_share', 'orders:cross_venue_repeat']
tr = by_class(O, on, cols, lab.y)
fig, ax = plt.subplots(figsize=(11, 3.4))
for j, c in enumerate(cols):
    ax.plot(range(len(LABELS)), tr[c].values, 'o-', label=c.split(':')[1], color=viz.CYCLE[j])
ax.set_xticks(range(len(LABELS)), LABELS); ax.set_xlabel('Titre'); ax.set_ylabel('part'); ax.legend(fontsize=7, ncol=3)
ax.set_title('Profil des parcours d\\'ordres par titre (train)'); viz.save(fig, 'd4_profiles')
"""),
('code', """
if y_test_pred is not None:
    te = by_class(lab.table(['orders'], 'test')[0], on, cols, y_test_pred)
    fig, axs = plt.subplots(1, len(cols), figsize=(13, 2.8))
    for a, c in zip(axs, cols):
        a.scatter(tr[c], te[c], s=14, color=viz.PAL['purple']); lim = [min(tr[c].min(), te[c].min()), max(tr[c].max(), te[c].max())]
        a.plot(lim, lim, 'k:', lw=.8); a.set_title(c.split(':')[1], fontsize=7); a.set_xlabel('train'); a.set_ylabel('test')
    fig.suptitle('Stabilité des parcours d\\'ordres : chaque point est un titre (sur la diagonale = stable)', fontweight='bold')
    viz.save(fig, 'd4_train_vs_test')
"""),
('code', FOOTER),
]

if __name__ == '__main__':
    for name, cells in [('V1-benchmark.ipynb', V1), ('V2-tree_models.ipynb', V2), ('V3-sequence_models.ipynb', V3),
                        ('V4-best_version_0507.ipynb', V4), ('V5-simple_improvements.ipynb', V5),
                        ('data_exploration/D1-anatomy_of_a_window.ipynb', D1), ('data_exploration/D2-stock_signatures.ipynb', D2),
                        ('data_exploration/D3-train_test_shift.ipynb', D3), ('data_exploration/D4-order_lifecycles.ipynb', D4)]:
        print(notebook(name, cells))
