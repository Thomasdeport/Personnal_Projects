# Tester des features vite, sans se mentir

## 1. Ajouter une feature : 3 minutes

```python
# cfm/features/mine.py
import numpy as np
from ..registry import block
from .common import share, nanmean

@block('cancel_speed', family='orders')
def cancel_speed(r):
    """Annulation rapide : ordre ajouté puis supprimé en ≤ 5 événements dans la fenêtre."""
    fast = (r.prev_action == 1) & (r.action == 2) & (r.gap <= 5)
    return {'fast_cancel_share': share(fast, r.action == 2),
            'fast_cancel_per_event': fast.mean(1)}
```

Ensuite, ajouter `mine` à `cfm/features/__init__.py`, puis :

```bash
python -m cfm screen univariate --config configs/base.json --blocks cancel_speed
python -m cfm screen forward    --config configs/base.json --blocks cancel_speed
```

Le bloc est calculé une fois, puis caché. Modifier la fonction change sa clé et ne recalcule que ce bloc.

Champs disponibles sur `r`, tous de forme (n, 100) :

- quantités brutes : `price`, `bid`, `ask`, `bq`, `aq`, `flux` ;
- catégories : `venue`, `action` (1 = A, 2 = D, 3 = U), `side` (1 = A, 2 = B), `trade`, `is_trade`, `oid` ;
- dérivées : `spread_t`, `mid`, `dmid_h`, `own_dist`, `depth`, `imbalance`, `flux_lots`, `occ`, `order_count`, `first_action`, `prev_action`, `gap`, `prev_venue`.

Règles d'un bloc :

1. **Jamais de label.** Jamais de statistique ajustée sur plusieurs fenêtres : pas de quantile global ni de moyenne du train. Ce qui s'ajuste va dans les scalers des modèles, calculés sur le fit.
2. **Rien que la fenêtre.** Une fenêtre est tronquée : une première apparition n'est pas une création, et un `gap` se compte en événements, pas en secondes.
3. **NaN plutôt que 0** quand la grandeur n'existe pas. XGBoost gère les NaN ; les réseaux les remplacent par la moyenne du fit.

## 2. L'entonnoir de screening

| Étape | Question | Coût | Décision |
|---|---|---|---|
| `univariate` | Ça sépare les titres (η² sur le fit) ? Ça dérive (KS train/test) ? C'est redondant (corrélation max avec la base) ? Constant ? | secondes | Écarter les colonnes constantes ou en double |
| `adversarial` | Ce bloc seul distingue-t-il train et test ? | ≈ 1 min / bloc | **Ne supprime rien.** Indique où chercher une représentation plus robuste |
| `forward` | Base + bloc, réentraîné : Δ valid (IC apparié) et Δ stress | ≈ 1 modèle / bloc | Garder si Δ valid > 0 avec borne basse > 0 **et** Δ stress ≥ 0 |
| `ablate` | Ensemble − bloc : le bloc paie-t-il encore sa place avec les autres ? | ≈ 1 modèle / bloc | Retirer les blocs dont le retrait ne coûte rien |
| `greedy` | Sélection pas à pas | k × modèles | Pratique, mais valid devient un score de sélection : reconfirmer sur une autre seed et sur stress |

Lecture de la carte utilité/drift (`figures/screen_map.png`) :

- en haut à gauche : informatif et stable, à garder ;
- en haut à droite : informatif mais instable, à garder et à rendre robuste (discrétiser, jitter, voir les augmentations) ;
- en bas : peu utile seul, mais peut l'être en interaction ; c'est le forward qui tranche.

Une AUC adversariale élevée **n'impose pas** de supprimer un bloc. Les versions « invariant » (0,469) et autres (0,449) suggèrent que retirer les niveaux coûte cher.

## 3. Plan d'expériences suggéré

Une différence à la fois, même split, même budget.

| # | Baseline → unique différence | Hypothèse | Abandon si |
|---|---|---|---|
| 0 | `prepare` puis lecture du `tick_report` et de `split.json` | Prix sur la grille de 0,01 ; test moins profond | Prix hors grille → rester en relatif |
| 1 | `forward` sur la base `base_relative + cat_freq` (≈ résumés actuels) | ticks / lots / orders / position battent les seuls résumés relatifs | Aucun bloc avec borne basse > 0 |
| 2 | XGB sur les blocs retenus vs MLP sur les mêmes blocs | Les arbres exploitent mieux les fréquences discrètes | — (sert de membre d'ensemble) |
| 3 | `control_gru` → `signature` | Tokens + conv + attention battent le BiGRU à entrée égale | Δ valid inférieur à l'écart entre 2 seeds |
| 4 | `signature` → `signature_no_bias` | Le biais même ordre apporte quelque chose | Δ ≈ 0 : le retirer (plus simple) |
| 5 | `signature` → `signature_robust` | Les augmentations (venue dropout, jitter de niveaux, dropout de blocs) réduisent l'écart valid/stress | Stress inchangé et valid en baisse |
| 6 | `signature` → `signature_relative_only` | Les niveaux absolus aident malgré le drift | Valid **et** stress égaux sans niveaux : préférer la version relative |
| 7 | Blend des membres retenus (moyenne) ; regarder accord et oracle | Erreurs complémentaires réseau / arbres | Accord > 0,9 et blend ≈ meilleur membre |

Pour les finalistes : au moins 2 seeds (`--set neural.train.seed=1`), en rapportant chaque seed. Comparer paramètres, minutes et nombre d'updates (`result.json`).

## 4. Pistes non implémentées, et pourquoi

- **Symétrie bid/ask comme augmentation.** Elle exige de connaître la convention de signe de `flux` : lié au côté, ou à l'ajout/retrait ? À vérifier dans les données avant de l'implémenter.
- **Recadrage de sous-fenêtres.** Incompatible avec l'embedding de position absolue, sauf à masquer.
- **Équilibrage des prédictions test** (`--balance`). Il est implémenté mais désactivé : il utilise l'ensemble du test non étiqueté. Vérifier le règlement et la répartition 81 600 = 24 × 3 400 avant usage. Le tenir hors du protocole « train only ».

## 5. Registre

`<lab>/ledger.csv` reçoit une ligne par screening, run, refit, soumission ou ouverture d'audit. Avant chaque soumission au leaderboard, noter dans `--note` la décision qu'elle doit trancher.
