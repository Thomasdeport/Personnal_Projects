# Changelog

## 0.6.1 — 2026-10-08 (papier : atlas et démonstration)
- Papier, 61 pages. Section 5, **atlas des données** : 14 figures réelles (fenêtres, tokens, ticks, lots, profondeur, dérive, vie des ordres). Section 17, **démonstration contrôlée** : 25 modèles sur un même protocole. L'échelle du CNN simple isole la représentation : **+16 points** à architecture égale (0,449 → 0,609), puis 0,701 avec toutes les étapes.
- `v_demo/figures/` : les figures clés des deux notebooks, avec un index.
- `make_vdemo.py` : correction de l'empreinte de tokens (colonnes constantes entre titres → NaN).

## 0.6.0 — 2026-10-07 (V5)
Après la V4 (LB **0,6551** pour C ; contrôle D sans élèves 0,6174 → les pseudo-labels apportent +3,8 pts).
- `CFM_V5.ipynb` : second tour d'auto-apprentissage. Le professeur est V4 C. Élèves `v5_q60` ×3 et `v5_q40` ×3 : même professeur, une seule différence, le quota. Refit, alignement, voisins, puis soumissions ALL / E / F (k=40). Introduction complète (contexte, parcours, hypothèses H1/H2, protocole écrit d'avance, pronostic).
- `cfm/report.py` : 13 figures de qualité rapport, en PNG 200 dpi et en PDF, avec un `index.md` des légendes. Elles couvrent :
  - le parcours LB et le schéma de la boucle ;
  - la sélection des pseudo-labels ;
  - les courbes d'apprentissage et les graines ;
  - la cascade des gains et la grille des voisins ;
  - la calibration, le rappel par titre et les confusions regroupées ;
  - les changements au test par rapport au professeur ;
  - la carte t-SNE.
- `teacher/v4_C_k40_probs.npz` (V4 C) et `teacher/v4_raw_probs.npz` (moyenne brute des 5 modèles V4, équilibrée) : non commités, à fournir via un dataset Kaggle (`TEACHER_DIR`).
- DEMO exécutée en 4 min (CPU, synthétique) : toutes les figures sont produites.

## 0.5.0 — 2026-10-07 (v_demo)
- Dossier **`v_demo/`** : deux notebooks complets, un seul « Run All » chacun, générés par `scripts/make_vdemo.py` et décrits dans `v_demo/README.md`.
  - `CFM_demo_models` : benchmark classique, arbres (familles de features une à une, LightGBM, CatBoost, importances, utilité/dérive, AUC adversariale), modèles séquentiels sur l'ancienne représentation, reconstruction simplifiée de l'hybride V2 (0,507), puis un petit CNN avec les améliorations ajoutées une par une (tokens ticks/lots, profondeur, durée, graines, Sinkhorn, voisins), et une synthèse de tous les modèles.
  - `CFM_demo_features` : anatomie d'une fenêtre, signatures des titres, dérive train/test, parcours d'ordres. Sans torch.
- Bibliothèque **`cfm/demo/`** (core, models, train, viz), courte et commentée. Le module de données n'importe pas torch.
- Bloc `ev_old` : l'ancienne représentation relative, pour des comparaisons à entrée égale.

## 0.4.0 — 2026-10-07 (V4)
Après la V3 (LB 0,6028 pour A, **0,6351** pour B, grâce au vote entre voisins).
- **Pseudo-étiquetage** (`neural.pseudo`) : des fenêtres test sont ajoutées à l'entraînement avec un poids réduit. Le fichier est vérifié par empreinte et par alignement sur les `obs_id` test. Le scaler reste ajusté sur les seules fenêtres étiquetées. La perte est pondérée par fenêtre.
- `transductive.select_pseudo` : une fenêtre n'est retenue que si la prédiction lissée (V3 B) et la prédiction brute (V3 A) concordent, avec un quota de 40 % par titre prédit pour ne pas hériter du biais de classes du professeur.
- `teacher/` : probabilités test de V3 A et B, envoyées sur Kaggle par `kaggle_upload.py`, jamais commitées.
- Configs `v4_xlong` (90 époques) et `v4_student` (`v3_long` + pseudo-labels, poids 0,5).
- `CFM_V4.ipynb` : élèves ×2, xlong ×3, ancre `v3_long` ×2 ; refit ; alignement ; grille de voisins ; soumissions C_k20/k40/k60 et D_k40.
- Un run ou un refit **terminé** avec la même configuration est réutilisé, même si le code a changé depuis (permet de reprendre `lab_v3`).
- `kaggle_upload.py` : passe à kagglehub la liste exacte des fichiers à ignorer (kagglehub applique `*.npz` à tous les chemins) ; n'exige plus un nom de notebook.
- 18 tests de contrat (+ pseudo-labels : fenêtres réellement entraînées, fichier modifié refusé).

## 0.3.0 — 2026-10-07 (V3)
Après le round 2 (LB 0,59999 : blend `sig2_depthaug` + `signature_no_bias`, équilibré).
- **Validation par grappes de régime** (`split.valid_mode='cluster'`) : k-means par titre sur profondeur, lots, ticks et venues ; des grappes entières vont en valid, pour approcher des jours non vus. Audit et stress sont inchangés (mêmes tirages).
- **EMA des poids** (`neural.train.ema`) : évaluation, sélection et refit sur les poids moyennés ; reprise exacte ; `refit_weights.pt` = poids utilisés pour le test.
- **Alignement de profondeur au test** (`predict_shifted`, `DepthShift`) : prédire avec les tailles × γ. γ est estimé sans labels à partir de l'écart de log-profondeur médiane ; le multiplicateur est choisi sur le stress.
- **Vote entre voisins** : nouvelle représentation `combo` (fenêtre ⊕ embeddings) ; pureté et grille écrites en CSV.
- Configs V3, une différence chacune vs `v3_base` (= `sig2_depthaug`) : `v3_nodepth`, `v3_depth05` (σ=0,5), `v3_ema`, `v3_long` (60 époques), `v3_big` (d=128, 3 convolutions, 681k paramètres).
- `CFM_V3.ipynb` : banc 6×2, finalistes 2×3 seeds, refit, alignement, voisins, soumissions A/B ; tous les tableaux de décision sont sauvegardés (`v3_*.csv`, `v3_decisions.json`).
- 17 tests de contrat (EMA + reprise, alignement γ=1 identique aux prédictions sauvegardées, refus des poids du refit sur partitions étiquetées, split par grappes).

## 0.2.0 — 2026-10-06
Round 2, à partir des résultats du round 1 (LB 0,57 brut, 0,5958 équilibré ; validation aléatoire 0,85 → non prédictive).
- `CFM_Round2.ipynb` : 6 configs × 2 seeds classées par **précision stress équilibrée** (règle écrite avant les résultats), indicateurs sans labels sur le test, finalistes + refit, lissage par voisins, deux soumissions au plus.
- Nouvelles configs, une différence chacune vs `signature_no_bias` : `sig2_nolevels` (sans tailles absolues), `sig2_depthaug` (profondeur × exp(N(0, 0,3))), `gru_wide` (BiGRU à capacité égale, ~332k paramètres).
- Augmentation `depth_scale` : même facteur sur log_bq/log_aq et sur les quantiles de profondeur du contexte, flux et lots inchangés.
- `SignatureNet.features` et `neural.train.embed` : représentations du modèle dev (valid/stress) ou du refit (test seulement, refus sur les partitions étiquetées).
- `cfm/transductive.py` : kNN exact (GPU si disponible), lissage par propagation, pureté des voisins, grille, application au test avec k × 4.
- `cfm/proxies.py` : confiance, entropie, équilibre et accord entre seeds sur le test.
- Blocs `ev_ticks` et `ev_sizes` (séparation prix en ticks / tailles en lots).
- Test local sur les probabilités du round 1 : le lissage dans l'espace des probabilités n'apporte rien après équilibrage (0,8281 → 0,8281 sur valid ∪ stress). Le round 2 teste donc des représentations de régime (fenêtre, embeddings).
- 4 tests de contrat ajoutés (15 au total).

## 0.1.2 — 2026-10-06
- `scripts/kaggle_upload.py` publie le code comme Dataset Kaggle privé versionné, via kagglehub, avec le commit git dans la note de version.
- Notebook : `CODE_DATASET` télécharge la dernière version du code ; il affiche la version utilisée.

## 0.1.1 — 2026-10-06
- Notebook unique à la racine, `CFM_Signature_Lab.ipynb` : tests, screening, 3 tabulaires, banc neuronal de 5 variantes (une différence à la fois), seeds des finalistes, blend et soumission.
- Les modèles d'arbres tournent en sous-processus (crash OpenMP xgboost/lightgbm + torch reproduit sur macOS).
- Version exécutée en DEMO : `notebooks/CFM_Signature_Lab_DEMO_executed.ipynb`.

## 0.1.0 — 2026-10-06
Première version, indépendante de V_init / Reaction R2 (ces bases restent intactes).

- Cache brut validé : contiguïté des fenêtres, unicité des `obs_id`, alignement des labels par `obs_id`, rang d'ordre par première apparition, estimation du tick avec contrôle de grille.
- Registre de features par blocs avec cache versionné ; 11 blocs fenêtre, 11 tokens, 2 blocs de canaux.
- Split : audit tiré en premier ; stress orienté automatiquement vers le test (corrige le `STRESS_TAIL='high'` codé en dur dans R1/R2, alors que le test est moins profond sur les figures 03/04).
- Screening : univarié, adversarial par bloc, forward / ablation réentraînés avec bootstrap apparié, sélection gloutonne.
- Modèles : XGBoost / LightGBM / MLP ; SignatureNet (tokens + conv dilatée + attention à biais « même ordre » + contexte) avec encodeurs alternatifs.
- Blend, soumission vérifiée, audit scellé, ledger.
- Tests de contrat sur données synthétiques. Aucune mesure sur les vraies données.
