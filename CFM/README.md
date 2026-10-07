# CFM Signature Lab

Identifier lequel des 24 titres a produit une fenêtre de 100 événements de carnet d'ordres, dans une période différente de celle du train.

Le repo est construit autour de trois idées :

1. **Unités naturelles.** Les prix et les quantités du challenge sont quantifiés : tick de 0,01, lots de 100. On exprime donc la donnée en ticks et en lots plutôt qu'en multiples d'une médiane locale. Le régime de tick et la convention de lot sont des signatures de titre.
2. **Événements comme mots.** Chaque événement devient un petit ensemble de tokens discrets : action, côté, trade, venue, distance au meilleur prix en ticks, taille en lots, spread en ticks, Δmid, occurrence de l'ordre, action précédente du même ordre. Les buckets sont fixés à l'avance, sans aucun seuil appris, donc sans fuite possible.
3. **Itérer vite et proprement sur les features.** On écrit une fonction, elle devient un bloc caché et versionné, puis elle passe un entonnoir de screening : univarié → adversarial → forward réentraîné → ablation. Chaque décision est consignée dans un ledger.

Rien dans ce repo n'a été exécuté sur les vraies données. Les tests et la démo tournent sur des données synthétiques, qui valident le **logiciel**, jamais la performance.

## Démarrage

### Kaggle (cible)

1. Créer un Dataset avec ce dossier, puis l'attacher à un notebook GPU avec les CSV du challenge.
2. Ouvrir **`CFM_Signature_Lab.ipynb`** (à la racine) et exécuter toutes les cellules : tests logiciels → préparation → screening → XGBoost/LightGBM/MLP → 5 variantes neuronales → seeds des finalistes → blend → soumission. Les CSV sont détectés par nom et par colonnes.
3. Récupérer `lab/submission_finalists_mean.csv` et `lab_results.zip` : ledger, configs, probabilités, figures.

**V4 :** `CFM_V4.ipynb` (réutilise `/kaggle/working/lab_v3` s'il existe ; environ 4 à 5 h de GPU, non mesuré) : pseudo-étiquetage à partir de V3 B, entraînement de 90 époques, variantes du vote entre voisins. Les fichiers `teacher/` doivent être présents dans le dataset Kaggle (`python scripts/kaggle_upload.py`).

**V3 :** `CFM_V3.ipynb` (nouveau lab `/kaggle/working/lab_v3`, environ 4 à 5 h de GPU, non mesuré) joue sur tous les leviers : validation par grappes, EMA, profondeur, capacité, durée, ensemble 2×3, alignement de profondeur, vote entre voisins. DEMO exécutée : `notebooks/CFM_V3_DEMO_executed.ipynb`.

**Round 2 :** `CFM_Round2.ipynb` (même mise en place, même `CODE_DATASET`) compare 6 configurations sans se fier à la validation aléatoire, puis teste le lissage par voisins. Version DEMO exécutée : `notebooks/CFM_Round2_DEMO_executed.ipynb`.

`notebooks/CFM_Signature_Lab_DEMO_executed.ipynb` montre le même notebook déjà exécuté sur données synthétiques (`DEMO=True`, CPU, ~2,5 min), pour voir les sorties sans rien lancer.

Coût mesuré en local (CPU 4 cœurs, données répliquées) : environ **3 min** pour calculer tous les blocs de features sur 242 400 fenêtres. La lecture du CSV, les modèles XGBoost et les réseaux n'ont **pas** été mesurés sur GPU Kaggle.

### Relier le code à Kaggle (kagglehub)

Le code est publié comme **Dataset Kaggle privé**. Chaque upload crée une nouvelle version, et le notebook télécharge la dernière.

1. Une seule fois : Kaggle → *Settings* → *API* → *Create New Token*. Placer le fichier dans `~/.kaggle/kaggle.json` puis faire `chmod 600 ~/.kaggle/kaggle.json`. Ne jamais le commiter.
2. Publier, depuis ce dossier :
   ```bash
   pip install kagglehub
   python scripts/kaggle_upload.py --dry-run       # liste ce qui partirait
   python scripts/kaggle_upload.py --notes "première version"
   ```
   Le handle par défaut est `<ton username>/cfm-signature-lab`. La note de version contient le commit git, et signale les modifications non commitées.
3. Dans le notebook Kaggle, renseigner `CODE_DATASET = '<ton username>/cfm-signature-lab'` dans la première cellule. Sinon, attacher le Dataset via *Add Input* et laisser `CODE_DATASET = None`. La première ligne affichée indique la version du code utilisée.

Ne sont jamais envoyés : `.git`, les caches, les données (`*.csv`, `*.npy`, `*.npz`), les poids (`*.pt`), les zips.

### En ligne de commande

```bash
pip install -r requirements.txt
python -m cfm prepare --config configs/base.json               # cache brut + tous les blocs + split
python -m cfm screen univariate --config configs/base.json
python -m cfm screen adversarial --config configs/base.json
python -m cfm screen forward --config configs/base.json
python -m cfm tabular --config configs/base.json --blocks base_relative cat_freq levels ticks orders --model xgb --refit
python -m cfm neural --config configs/signature.json --name sig_s0 --refit epochs
python -m cfm neural --config configs/control_gru.json --name gru_s0 --refit epochs
python -m cfm blend --config configs/base.json --runs sig_s0 gru_s0 tab_xgb_xxxx --submit
```

Toute valeur de config se surcharge en ligne : `--set neural.train.lr=5e-4 --set device=cpu`.

### Démo et tests en local (CPU, synthétique)

```bash
python -m cfm synthetic --out data/synth
python tests/test_lab.py            # ou : pytest -q tests
```

## Architecture du code

| Fichier | Rôle |
|---|---|
| `cfm/io.py` | CSV → cache brut validé : fenêtres contiguës de 100 lignes, `obs_id` uniques, labels alignés par `obs_id`, catégories avec modalité inconnue = 0, `order_id` remplacé par son rang de première apparition (égalités conservées, distances numériques jetées), estimation et vérification du tick |
| `cfm/registry.py` | Décorateur `@block`, cache par bloc (clé = source de la fonction + helpers + tick/lot + cache brut), assemblage des matrices |
| `cfm/features/common.py` | Grandeurs partagées en ticks/lots : spread, Δmid en demi-ticks, distance au meilleur prix de son côté, profondeur, parcours d'ordre vectorisés (occurrence, action précédente, écart, première action) |
| `cfm/features/window.py` | Blocs par fenêtre : `base_relative` (≈ résumés actuels), `cat_freq`, `cat_cross`, `levels`, `ticks`, `lots`, `position`, `orders`, `book_dyn`, `tokens_bow`, `transitions` |
| `cfm/features/events.py` | Tokens par événement et canaux continus pour le réseau |
| `cfm/split.py` | Partitions par classe : audit tiré en premier, stress = queue de profondeur orientée vers le test (`auto`), valid, fit |
| `cfm/screen.py` | Entonnoir : `univariate`, `adversarial`, `forward`, `ablate`, `greedy` |
| `cfm/tabular.py` | XGBoost (GPU), LightGBM, MLP torch : même interface, early stopping sur un sous-ensemble interne du fit |
| `cfm/neural/` | SignatureNet, données sur GPU, augmentations, entraînement avec reprise, refit |
| `cfm/blend.py` | Moyenne ou poids appris sur valid, accord entre modèles, oracle, équilibrage Sinkhorn (transductif, désactivé par défaut), soumission vérifiée |
| `cfm/audit.py` | Ouverture unique de l'audit, avec un marqueur irréversible |
| `cfm/plots.py` | Figures, chacune répondant à une question |

## SignatureNet

```
tokens (B,100,11) ─ Σ embeddings ─┐
cont   (B,100,11) ─ linéaire ─────┼─ + position ─ LN ─ 2× conv dilatée résiduelle ─ 2× attention + β·[même ordre] ─ pool(attn, moy, max) ─┐
ctx    (B,~80)    ─ MLP ──────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─ fusion ─ 24
```

- **Convolutions dilatées** : motifs locaux (rafales d'annulations, ping-pong bid/ask) sur un champ de 13 événements consécutifs (noyau 5, dilatations 1 et 2).
- **Attention à biais « même ordre »** : chaque tête apprend un scalaire β_h ajouté aux scores entre événements du même ordre. Avec β = 0, on retrouve une attention standard. L'apport du biais se teste avec `configs/signature_no_bias.json`.
- **Contexte** : blocs tabulaires standardisés sur le fit seulement.
- **Encodeurs disponibles** (`neural.encoder`) : `conv`, `gru`, `conv_gru`, `conv_rel`, `rel`. Le contrôle `control_gru.json` est proche de R2 : BiGRU et fusion linéaire, mêmes entrées.

## Protocole

- `valid` choisit les époques, les blocs et les poids de blend. Il devient donc un score **de sélection** dès qu'on s'en sert pour choisir.
- `stress` est la queue de profondeur médiane, orientée vers le test (`stress_tail: auto`, qui utilise les covariables test non étiquetées, consigné dans `split.json`). Ce n'est **pas** une période future.
- `audit` est tiré en premier, avec une seed dédiée. Il est identique quel que soit le réglage du stress et n'est lu que par `cfm.audit.open_audit(confirm=True)`.
- Le refit utilise tous les labels, audit compris. Il ne produit aucun score indépendant. Deux budgets sont possibles : `epochs` (même calendrier de LR par époque, plus d'updates) ou `steps` (même nombre d'updates que le développement).
- Les intervalles bootstrap rééchantillonnent les fenêtres comme si elles étaient indépendantes. Ce n'est pas le cas (20 fenêtres par titre et par jour) : ces intervalles sont **trop étroits**.

Voir `EXPERIMENTS.md` pour ajouter une feature et pour le plan d'expériences.

## Dépannage

- **macOS : segfault ou blocage** quand XGBoost/LightGBM et torch tournent dans le même processus (deux runtimes OpenMP, reproduit : code 139). Le notebook lance donc toujours les modèles d'arbres dans un sous-processus (`cli(...)`). En script, garder ces familles dans des processus séparés.
- **`device="cuda"` sans GPU** : erreur explicite, jamais de repli silencieux. Passer `--set device=cpu` volontairement.
- **« Split already exists with another config »** : le split est figé par `lab_dir`. Changer de `lab_dir` pour en tester un autre.
# CFM2026
