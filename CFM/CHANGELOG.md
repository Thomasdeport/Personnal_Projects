# Changelog

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
