# Changelog

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
