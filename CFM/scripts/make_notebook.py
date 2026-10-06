"""Generate CFM_Signature_Lab.ipynb at the repo root (run from the root: python scripts/make_notebook.py)."""
from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# CFM Signature Lab — banc d'essai complet

Ce notebook lance **tout** dans l'ordre. Chaque étape lourde est cachée sur disque : si la session s'arrête,
relancer le notebook reprend là où il s'était arrêté.

| § | Étape | Ce qu'on décide |
|---|---|---|
| 0 | Tests logiciels (synthétique) | Le code est-il sain avant de dépenser du GPU ? |
| 1 | Préparation + audit des unités | Tick, lots, catégories inconnues, sens du stress |
| 2 | Screening des features | Quels blocs garder (règle écrite **avant** de regarder) |
| 3 | Modèles tabulaires | XGBoost, LightGBM, MLP sur les blocs retenus |
| 4 | Banc neuronal | 5 variantes, **une seule différence** à chaque fois |
| 5 | Seeds des finalistes | Dispersion avant de conclure |
| 6 | Blend + soumission | Moyenne des finalistes, refit sur tous les labels |
| 7 | Audit | Une seule fois, décisions gelées (désactivé par défaut) |

Partitions : `fit` entraîne, `valid` sélectionne, `stress` = queue de profondeur orientée vers le test (diagnostic),
`audit` scellé. Aucun résultat synthétique (`DEMO=True`) n'a de valeur de performance.
""")
code("""
import os, sys, json, subprocess
from pathlib import Path
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')   # macOS : xgboost + torch dans un même processus

def find_repo():
    for p in [Path('.'), Path('..'), Path('/kaggle/working/cfm-signature-lab')]:
        if (p / 'cfm' / '__init__.py').exists():
            return p.resolve()
    for p in Path('/kaggle/input').glob('**/cfm/__init__.py'):
        return p.parent.parent
    raise FileNotFoundError('Dossier du repo introuvable : attacher le Dataset contenant cfm/')

REPO = find_repo(); sys.path.insert(0, str(REPO))

DEMO = False        # True : données synthétiques + CPU, ~2 min. Vérifie le logiciel, jamais la performance.
RUN_TESTS = True    # § 0 : ~3-4 min CPU
SEEDS_SCREEN = [0]  # seed des comparaisons
SEEDS_FINAL = [1]   # seeds supplémentaires pour les finalistes
N_FINALISTS = 2

from cfm import config, io, registry, split, screen, blend, plots   # pas d'xgboost/lightgbm dans ce processus
from cfm.__main__ import prepare
from cfm.neural.train import fit, refit
import pandas as pd
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)

LAB = 'demo_lab' if DEMO else '/kaggle/working/lab'
DEMO_SET = ['neural.d_model=32', 'neural.heads=2', 'neural.train.epochs=4', 'neural.train.patience=4',
            'neural.train.batch_size=64', 'neural.train.amp=false'] if DEMO else []

def load_cfg(name='base', extra=()):
    over = [f'lab_dir={LAB}'] + (['device=cpu'] + DEMO_SET if DEMO else []) + list(extra)
    return config.load(REPO / 'configs' / f'{name}.json', over)

cfg = load_cfg()
if DEMO:
    cfg['screen']['fit_frac'] = 1.0

def cli(*args, name='base'):
    # `python -m cfm ...` in a separate process. Tree models (XGBoost/LightGBM) always run this way:
    # mixing their OpenMP runtime with torch in one process can crash (observed on macOS).
    over = [f'lab_dir={LAB}'] + (['device=cpu', 'screen.fit_frac=1.0'] + DEMO_SET if DEMO else [])
    over += [f'screen.seeds={json.dumps(SEEDS_SCREEN)}']
    cmd = [sys.executable, '-u', '-m', 'cfm', *map(str, args), '--config', str(REPO / 'configs' / f'{name}.json')]
    cmd += sum([['--set', o] for o in over], [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, 'PYTHONPATH': str(REPO)})
    lines = []
    for line in p.stdout:
        lines.append(line)
        if not line.startswith('[block'):
            print(line, end='')
    if p.wait():
        raise RuntimeError(''.join(lines[-30:]))
print('repo', REPO, '| lab', LAB, '| device', cfg['device'])
""")
md("## 0. Tests logiciels\nAlignement labels/`obs_id`, fenêtres contiguës, invariance au renumérotage des ordres, unités en ticks, cache, split, reprise, soumission.")
code("""
if RUN_TESTS:
    out = subprocess.run([sys.executable, str(REPO / 'tests' / 'test_lab.py')], capture_output=True, text=True,
                         env={**os.environ, 'OMP_NUM_THREADS': '1'})
    print('\\n'.join(l for l in out.stdout.splitlines() if l.startswith(('PASS', 'FAIL')) or 'failure' in l))
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
""")
md("## 1. Préparation et audit des unités")
code("""
files = None
if DEMO:
    from cfm.synthetic import create
    files = create('demo_data')
sp = prepare(cfg, files)          # sinon : détection auto dans /kaggle/input ; sinon renseigner cfg['files']
meta = io.meta(LAB)
print(json.dumps({k: meta[k] for k in ['tick_report', 'unknown_codes', 'venue_vocab', 'counts']}, indent=2))
print(json.dumps(json.loads((Path(LAB) / 'split.json').read_text()), indent=2))
assert meta['tick_report'].get('grid_ok', True), 'Prix hors grille : les blocs en ticks sont approximatifs, lire EXPERIMENTS.md'
registry.catalogue()
""")
md("""
## 2. Screening des features
Règle de sélection fixée **avant** de lire les résultats : un bloc est gardé si, ajouté à la base et réentraîné,
la borne basse de l'IC bootstrap de Δ valid est > 0 pour toutes les seeds, et Δ stress ≥ −1 point.
(IC optimiste : les fenêtres d'un même titre-jour ne sont pas indépendantes.)
""")
code("""
S = cfg['screen']
uni = screen.univariate(LAB, cfg, S['base'] + S['candidates'], base=S['base'])
plots.screen_map(uni, LAB)
uni.head(25)
""")
code("""
cli('screen', 'adversarial')
adv = pd.read_csv(Path(LAB) / 'screen_adversarial.csv')
plots.adversarial_bars(adv, LAB)
adv.sort_values('auc_train_vs_test', ascending=False)
""")
code("""
cli('screen', 'forward')
fwd = pd.read_csv(Path(LAB) / 'screen_forward.csv')
plots.forward_bars(fwd, LAB)
cand = fwd[fwd.candidate != '(base)']
ok = cand.groupby('candidate').apply(lambda d: (d.d_valid_lo > 0).all() and (d.d_stress_acc >= -0.01).all())
KEEP = S['base'] + [c for c in S['candidates'] if ok.get(c, False)]
print('KEEP =', KEEP)
cand.sort_values('d_valid_acc', ascending=False)
""")
code("""
extra = [b for b in KEEP if b not in S['base']]
if extra:
    cli('screen', 'ablate', '--blocks', *extra)
    abl = pd.read_csv(Path(LAB) / 'screen_ablate.csv')
    display(abl)
else:
    print('Aucun bloc ajouté à la base : pas d’ablation à faire.')
""")
md("## 3. Modèles tabulaires sur les blocs retenus")
code("""
def run_tab(model, name, seed):
    if not (Path(LAB) / 'runs' / name / 'test_refit.npz').exists():
        cli('tabular', '--blocks', *KEEP, '--model', model, '--name', name, '--seed', seed, '--refit', '--note', 'blocs KEEP')
    return json.loads((Path(LAB) / 'runs' / name / 'result.json').read_text())

tab = {f'tab_{m}': run_tab(m, f'tab_{m}', 0) for m in ['xgb', 'lgbm', 'mlp']}
pd.DataFrame(tab.values())[['name', 'n_features', 'best_iter', 'seconds', 'valid_acc', 'valid_logloss', 'stress_acc', 'stress_logloss']]
""")
md("""
## 4. Banc neuronal — une différence à la fois

| Run | Différence | Question |
|---|---|---|
| `control_gru` | BiGRU + fusion linéaire (proche R2) | référence |
| `signature` | tokens + conv + attention à biais même ordre | l'architecture apporte-t-elle quelque chose à entrée égale ? |
| `signature_no_bias` | β même ordre désactivé | le biais relationnel sert-il ? |
| `signature_robust` | venue dropout, jitter des niveaux, dropout de blocs | réduit-on l'écart valid → stress ? |
| `signature_relative_only` | sans niveaux absolus | les niveaux aident-ils malgré le drift ? |
""")
code("""
EXPERIMENTS = ['control_gru', 'signature', 'signature_no_bias', 'signature_robust', 'signature_relative_only']
neural = {}
for exp in EXPERIMENTS:
    for seed in SEEDS_SCREEN:
        name = f'{exp}_s{seed}'
        neural[name] = fit(LAB, load_cfg(exp, [f'neural.train.seed={seed}']), name, note=f'banc § 4 : {exp}')
res = pd.DataFrame(neural.values()).set_index('name')
ref = res.loc[f'signature_s{SEEDS_SCREEN[0]}']
res['Δvalid_vs_signature'] = res.valid_acc - ref.valid_acc
res['Δstress_vs_signature'] = res.stress_acc - ref.stress_acc
res[['encoder', 'params', 'best_epoch', 'epochs_run', 'minutes', 'valid_acc', 'valid_logloss', 'stress_acc',
     'stress_logloss', 'Δvalid_vs_signature', 'Δstress_vs_signature']]
""")
code("""
plots.histories(LAB, list(neural))
plots.recall_by_class(LAB, f'signature_s{SEEDS_SCREEN[0]}')
""")
md("## 5. Finalistes : seeds supplémentaires\nLes `N_FINALISTS` meilleurs runs (valid) toutes familles confondues, relancés avec d'autres seeds. On rapporte chaque seed.")
code("""
allres = pd.concat([pd.DataFrame(tab.values()).set_index('name'), res], sort=False)
finalists = allres.sort_values('valid_acc', ascending=False).index[:N_FINALISTS].tolist()
print('finalistes :', finalists)
seed_rows = []
for f in finalists:
    for seed in SEEDS_FINAL:
        if f.startswith('tab_'):
            r = run_tab(f.split('_')[1], f'{f}_s{seed}', seed)
        else:
            exp = f.rsplit('_s', 1)[0]
            r = fit(LAB, load_cfg(exp, [f'neural.train.seed={seed}']), f'{exp}_s{seed}', note='seed finaliste')
        seed_rows.append({'family': f, 'run': r['name'], 'seed': seed, 'valid_acc': r['valid_acc'], 'stress_acc': r['stress_acc']})
seeds = pd.concat([pd.DataFrame([{'family': f, 'run': f, 'seed': int(allres.loc[f].get('seed', 0) or 0),
                                  'valid_acc': allres.loc[f].valid_acc, 'stress_acc': allres.loc[f].stress_acc} for f in finalists]),
                   pd.DataFrame(seed_rows)])
display(seeds)
seeds.groupby('family')[['valid_acc', 'stress_acc']].agg(['mean', 'std', 'count'])
""")
md("## 6. Blend des finalistes, refit, soumission\nRègle : moyenne simple de tous les runs des finalistes (toutes seeds). Pas de poids appris → valid reste un score propre du blend.")
code("""
MEMBERS = seeds.run.tolist()
table, w, agree, oracle = blend.report(LAB, MEMBERS, optimise=False)
plots.agreement(agree, LAB)
print('oracle (au moins un membre juste sur valid) :', round(oracle, 4))
table
""")
code("""
for m in MEMBERS:
    if not m.startswith('tab_'):                   # les tabulaires ont déjà été refittés
        exp = m.rsplit('_s', 1)[0]; seed = int(m.rsplit('_s', 1)[1])
        refit(LAB, load_cfg(exp, [f'neural.train.seed={seed}']), m, 'epochs')
path = blend.submission(LAB, MEMBERS, None, source='test_refit', balance=False, name='submission_finalists_mean')
print(path); pd.read_csv(path).head()
# Variante transductive, seulement si le règlement l'autorise :
# blend.submission(LAB, MEMBERS, None, source='test_refit', balance=True, name='submission_finalists_balanced')
""")
code("""
import zipfile
dest = Path(LAB).parent / ('demo_results.zip' if DEMO else 'lab_results.zip')
with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in Path(LAB).rglob('*'):
        if p.is_file() and not {'raw', 'features'} & set(p.relative_to(LAB).parts) and p.suffix != '.pt':
            z.write(p, p.relative_to(LAB))
print(dest, round(dest.stat().st_size / 1e6, 1), 'Mo — ledger, configs, probabilités, figures, soumissions')
pd.read_csv(Path(LAB) / 'ledger.csv').tail(15)
""")
md("## 7. Audit — une seule fois\nÀ décommenter seulement quand blocs, modèles, seeds, budget de refit et règle de blend sont figés.")
code("""
# from cfm.audit import open_audit
# open_audit(LAB, MEMBERS, confirm=True)
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}})
out = Path(__file__).resolve().parents[1] / 'CFM_Signature_Lab.ipynb'
nbf.write(nb, out)
print(out)
