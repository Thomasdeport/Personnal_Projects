# v_demo — comprendre le projet en neuf notebooks courts

Ce dossier raconte le projet **sans le pipeline complet** : des modèles simples, chacun entraîné en moins de 20 minutes sur GPU, avec un protocole identique partout et une figure par question.

| Notebook | Question | Durée GPU (estimée) |
|---|---|---|
| `V1-benchmark` | Que vaut-on sans rien d'intelligent ? (hasard, majorité, Naive Bayes, régression logistique, kNN, forêt aléatoire) | 5–10 min |
| `V2-tree_models` | Les features du papier aident-elles **le même** XGBoost ? LightGBM, CatBoost, importances, carte utilité/dérive, AUC adversariale | 10–20 min |
| `V3-sequence_models` | Lire la séquence plutôt que des statistiques : MLP, CNN, GRU, petit Transformer, sur l'**ancienne** représentation | ~15 min |
| `V4-best_version_0507` | Reconstruction simplifiée du meilleur modèle d'avant (Transformer hybride V2, 0,507 au LB) | ~15 min |
| **`V5-simple_improvements`** | **Le cœur : un petit CNN, puis chaque amélioration du papier ajoutée une à une** | 15–20 min |
| `data_exploration/D1-anatomy_of_a_window` | Une vraie fenêtre : carnet, événements, parcours d'ordres, tokens | < 2 min |
| `data_exploration/D2-stock_signatures` | Ce qui distingue les titres : ticks, lots, venues, profondeur, empreinte de tokens | < 3 min |
| `data_exploration/D3-train_test_shift` | Ce qui change entre train et test ; hypothèse « prix plus élevés » | ~5 min |
| `data_exploration/D4-order_lifecycles` | Comment vivent les ordres, titre par titre | < 2 min |

## Lancer sur Kaggle

1. Publier le code (dossier `CFM`, `teacher/` compris) : `python scripts/kaggle_upload.py --notes "v_demo"`.
2. Ouvrir un notebook de `v_demo/` dans Kaggle (GPU), attacher les CSV du challenge, renseigner `CODE_DATASET` dans la première cellule, puis « Run All ».
3. Le **premier** notebook lancé construit le cache dans `/kaggle/working/lab_v3` (≈ 20 min, une seule fois par session). Les suivants le réutilisent. Si le lab de la V3 ou de la V4 est encore là, rien n'est recalculé.
4. Les figures sont écrites dans `figures/<notebook>/` et zippées à la fin de chaque notebook.

`DEMO = True` dans la première cellule fait tourner un notebook en quelques minutes sur CPU, avec des données synthétiques. Ça vérifie la mécanique, pas la performance.

## Le protocole, commun à tous les notebooks

- **fit** entraîne ; la standardisation est estimée sur le fit seulement.
- **valid** choisit l'époque. Ce sont des grappes de régimes tenues à l'écart : plus dur qu'un tirage aléatoire, mais encore optimiste par rapport au leaderboard.
- **stress** (les fenêtres aux carnets les moins profonds, comme le test) est un diagnostic.
- La précision **équilibrée** applique la correction de Sinkhorn (explication dans V5).

## La bibliothèque

`cfm/demo/` (≈ 500 lignes commentées) : `core.py` (données), `models.py` (modèles), `train.py` (boucle d'entraînement), `viz.py` (figures). Elle est indépendante du pipeline principal et faite pour être lue.

Les notebooks sont générés par `python scripts/make_vdemo.py` et ne sont pas suivis par git.
