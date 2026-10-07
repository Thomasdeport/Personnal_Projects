# Registre des soumissions

Une ligne par soumission : quel fichier, quelle décision elle devait trancher, et ce qu'on en a conclu.
Le leaderboard public est la seule mesure hors période dont on dispose. On le consulte avec parcimonie.

| Date | Fichier (sha256, 16 premiers caractères) | Contenu | Valid (blend) | Stress (blend) | **Public LB** | Question posée → conclusion |
|---|---|---|---|---|---|---|
| — | ancien record (V2, ensemble Transformer hybride) | — | — | — | 0,5072 | Référence historique |
| 2026-10-06 | `submission_finalists_mean.csv` (`085c678bb63df827`) | Moyenne de `signature_s0`, `signature_s1`, `signature_robust_s0`, `signature_robust_s1`, refit sur tous les labels (budget : époques) | 0,852 | 0,741 | **0,57** | Les tokens en ticks/lots et SignatureNet transfèrent-ils hors période ? → **Oui : +6,3 pts sur l'ancien record.** Mais l'écart validation → LB est de 28 pts : la validation aléatoire ne sert pas d'estimation du test |
| 2026-10-06 | `submission_finalists_mean_BALANCED.csv` (`ec28beb9c190f260`) | Mêmes probabilités que la ligne précédente, équilibrage Sinkhorn sur le test (transductif : utilise X_test sans labels) | 0,860 | 0,806 | **0,5958** | Le décalage de prior vers les titres peu profonds coûte-t-il des points ? → **Oui : +2,6 pts.** Le test est probablement équilibré entre titres. Le gain est inférieur à celui du stress (+6,4) : le stress n'imite le test qu'en partie. Conformité au règlement à confirmer pour le classement final |
| 2026-10-06 | `submission_r2_balanced.csv` (`132004a65595e6a9`) | Round 2 : moyenne de `sig2_depthaug_s0/s1` + `signature_no_bias_s0/s1`, refit, équilibrage Sinkhorn | 0,859 | 0,814 | **0,59999** | La mise à l'échelle de la profondeur (hypothèse « prix plus élevés ») transfère-t-elle ? → **Oui, mais peu : +0,4 pt au LB** (contre +0,9 à +1,9 en stress). 14,8 % des prédictions test ont changé pour un solde net faible : le gain est réel en direction, mais de l'ordre du bruit entre runs. Le stress, décalé en profondeur seulement, surestime l'effet d'une augmentation de profondeur |
| 2026-10-07 | `submission_v3_A.csv` (`ceeed4275737bbd5`) | V3 : `v3_long` ×3 + `v3_ema` ×3 seeds, refit, tailles du test × γ=1,154 (alignement de profondeur), équilibrage | 0,812 (valid **par grappes**) | 0,827 | **?** | 60 époques + EMA + alignement de profondeur font-ils mieux que 0,59999 ? 14,8 % des prédictions diffèrent |
| 2026-10-07 | `submission_v3_B.csv` (`1d3792cf495ef80d`) | A + vote entre voisins (représentation combo, k_test=40, α=0,5, 5 itérations) | — | pool +1,3 pt | **?** | Les voisins (embeddings + régime) corrigent-ils des erreurs au test ? 9,1 % des prédictions diffèrent de A |

## Lecture

- L'écart validation → LB de 0,85 → 0,57 confirme la dépendance intra-jour : 20 fenêtres par titre et par jour, réparties entre fit et valid.
- Le stress (0,74) est lui aussi optimiste, mais moins.
- L'équilibrage aide sur le stress ET sur le LB : sur ce point, le stress a correctement prédit le sens de l'effet.
- Round 2 : le stress a de nouveau prédit le bon sens (3/3), mais il surestime systématiquement l'ampleur.
- Tant qu'on n'a pas une validation par groupes, comparer les modèles se fait avec le stress, les indicateurs sans labels sur le test (accord entre seeds, confiance, équilibre des prédictions) et quelques soumissions bien choisies.
