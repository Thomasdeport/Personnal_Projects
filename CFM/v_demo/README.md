# v_demo — le projet en deux notebooks

Deux notebooks, un seul « Run All » chacun, avec un protocole identique partout. Chaque modèle s'entraîne en moins de 20 min sur GPU.

## `CFM_demo_models.ipynb` — tous les modèles (≈ 1 h à 1 h 20 sur GPU, non mesuré)

| Partie | Contenu |
|---|---|
| 1 · Benchmark | Hasard, majorité, Naive Bayes, régression logistique, kNN, forêt aléatoire, sur les statistiques de la première version |
| 2 · Arbres | XGBoost avec les familles de features ajoutées une à une, LightGBM, CatBoost, importances, carte utilité/dérive, AUC adversariale |
| 3 · Séquences | MLP, CNN, GRU, petit Transformer, sur l'**ancienne** représentation |
| 4 · Le 0,507 | Reconstruction simplifiée du Transformer hybride V2, avec une soumission de contrôle |
| 5 · Améliorations | Un petit CNN (≈ 100 k paramètres), puis : tokens ticks/lots → profondeur → durée → 3 graines → Sinkhorn → vote entre voisins. Cascade, scores publics réels pour comparaison, confusions, carte t-SNE, soumissions par étape |
| Synthèse | Tous les modèles sur un même graphique (`synthese_modeles.csv`) |

## `CFM_demo_features.ipynb` — les données et les features (≈ 10 min)

| Partie | Contenu |
|---|---|
| 1 · Anatomie | Une vraie fenêtre (titre à gros tick contre titre à petit tick), parcours d'ordres, tokens |
| 2 · Signatures | Ticks, lots, venues, position et profondeur par titre, train contre test ; empreinte de tokens |
| 3 · Dérive | Carte utilité/dérive, colonnes les plus décalées, profondeur contre odd lots, AUC adversariale |
| 4 · Ordres | Transitions A/D/U par titre, profils de parcours, stabilité train/test |

Le test n'a pas de labels : pour le découper par titre, on utilise le titre **prédit** par V3 B, si `teacher/v3_B_probs.npz` est présent dans le code. Sinon, les barres « test » sont masquées.

## Lancer sur Kaggle

1. Attacher le code (dataset) et les CSV du challenge, activer le GPU.
2. Renseigner `CODE_DATASET` dans la première cellule de code, puis « Run All ».
3. Le cache est construit dans `/kaggle/working/lab_v3` lors du premier notebook (≈ 20 min). Il est réutilisé par le second, et par V3/V4 s'il existe déjà.
4. Les figures sont enregistrées dans `figures/models/` et `figures/features/`, puis zippées à la fin.

`DEMO = True` : données synthétiques sur CPU, quelques minutes. Ça vérifie la mécanique, les scores n'ont aucune valeur.

## Le protocole

- **fit** entraîne ; la standardisation est estimée sur le fit seulement.
- **valid** (grappes de régimes tenues à l'écart) choisit l'époque.
- **stress** (carnets les moins profonds, comme le test) est un diagnostic.
- La précision **équilibrée** applique la correction de Sinkhorn.

Les notebooks sont générés par `python scripts/make_vdemo.py` (non suivis par git). La bibliothèque est dans `cfm/demo/` : courte, commentée, indépendante du pipeline principal.
