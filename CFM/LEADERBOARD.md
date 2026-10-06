# Registre des soumissions

Une ligne par soumission : quel fichier, quelle décision elle devait trancher, et ce qu'on en a conclu.
Le leaderboard public est la seule mesure hors période dont on dispose. On le consulte avec parcimonie.

| Date | Fichier (sha256, 16 premiers caractères) | Contenu | Valid (blend) | Stress (blend) | **Public LB** | Question posée → conclusion |
|---|---|---|---|---|---|---|
| — | ancien record (V2, ensemble Transformer hybride) | — | — | — | 0,5072 | Référence historique |
| 2026-10-06 | `submission_finalists_mean.csv` (`085c678bb63df827`) | Moyenne de `signature_s0`, `signature_s1`, `signature_robust_s0`, `signature_robust_s1`, refit sur tous les labels (budget : époques) | 0,852 | 0,741 | **0,57** | Les tokens en ticks/lots et SignatureNet transfèrent-ils hors période ? → **Oui : +6,3 pts sur l'ancien record.** Mais l'écart validation → LB est de 28 pts : la validation aléatoire ne sert pas d'estimation du test |
| à faire | `submission_finalists_mean_BALANCED.csv` | Mêmes probabilités, équilibrage Sinkhorn sur le test (transductif) | 0,860 | 0,806 | ? | Le décalage de prior vers les titres peu profonds coûte-t-il des points ? **À soumettre seulement si le règlement autorise l'usage de X_test.** |

## Lecture

- L'écart validation → LB de 0,85 → 0,57 confirme la dépendance intra-jour : 20 fenêtres par titre et par jour, réparties entre fit et valid.
- Le stress (0,74) est lui aussi optimiste, mais moins.
- Tant qu'on n'a pas une validation par groupes, comparer les modèles se fait avec le stress, les indicateurs sans labels sur le test (accord entre seeds, confiance, équilibre des prédictions) et quelques soumissions bien choisies.
