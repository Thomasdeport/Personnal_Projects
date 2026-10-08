# Figures clés des notebooks de démonstration

Résultats réels (Kaggle, GPU) des deux notebooks `v_demo/`. Toutes ces figures sont aussi commentées dans le papier : section 5 (atlas des données) et section 17 (démonstration contrôlée).

## `models/` : 25 modèles, un seul protocole

| Figure | Ce qu'elle montre |
|---|---|
| `synthese.png` | Les 25 modèles classés sur valid ∪ stress : le CNN simple à tokens (0,701) dépasse la reconstruction de l'ancien record (0,637) |
| `v5_ladder.png` | **La figure clé** : même petit CNN, une amélioration à la fois. Tokens ticks/lots **+16 pts**, profondeur +1,0, entraînement long +3,9, 3 graines +2,5, Sinkhorn +1,1, voisins +0,7 |
| `v5_curves.png` | Apprentissage : l'ancienne représentation plafonne vers 0,47 |
| `v5_confusion_before.png`, `v5_confusion_after.png` | Confusions avant / après équilibrage et voisins |
| `v5_recall.png` | Rappel par titre : le gain est général |
| `v5_tsne.png` | Représentations : le test (gris) occupe les mêmes régions que la valid |
| `v2_compare.png` | Arbres : chaque famille stable fait monter le stress, les features de profondeur le font s'effondrer (piège de régime) |
| `v2_importance.png` | XGBoost s'appuie sur le spread en ticks… et sur la profondeur |
| `v1_compare.png`, `v1_confusion.png` | Benchmark classique sur les premières statistiques (≤ 0,25) |
| `v3_compare.png`, `v3_curves.png` | Lire la séquence, même avec l'ancienne représentation : MLP 0,245 → Transformer 0,569 |
| `v4_compare.png`, `v4_curves.png` | Reconstruction de l'hybride V2 (0,507 au LB) : son stress (0,52) prédit bien son score public |
| `synthese_modeles.csv` | Les chiffres des parties 1 à 4 |

## `features/` : atlas des données

| Figure | Ce qu'elle montre |
|---|---|
| `d1_window_big.png`, `d1_window_small.png` | Une vraie fenêtre, titre à gros tick contre titre à petit tick |
| `d1_tokens.png` | Une fenêtre telle que le réseau la lit (11 tokens par événement) |
| `d1_orders.png` | ~70 ordres par fenêtre ; 30 % apparaissent d'abord par une suppression |
| `d2_ticks.png` | Régime de spread par titre : la signature stable (train ≈ test) |
| `d2_lots.png`, `d2_depth.png` | Lots et profondeur : séparateurs mais décalés sur le test |
| `d2_position.png`, `d2_venues.png` | Distance au meilleur prix et venues : signatures secondaires stables |
| `d2_token_fingerprint.png` | Empreinte « sac de mots » des titres, regroupés par ressemblance |
| `d3_top_shifts.png`, `d3_depth_vs_oddlots.png` | Ce qui dérive : moins de profondeur, plus d'odd lots (prix plus élevés ?) |
| `d4_profiles.png`, `d4_train_vs_test.png` | La vie des ordres : stable d'une période à l'autre |

Pour les barres « test », le titre utilisé est celui **prédit** par V3 B (0,635 au LB) : le test n'a pas de labels.
