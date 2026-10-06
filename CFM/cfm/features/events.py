"""Event-level inputs for the neural encoder.

Tokens: fixed, data-free buckets (no threshold fitted on any population → no leakage, no drift in the
vocabulary). Index 0 is always "invalid/unknown/none".
Event channels: a few continuous values, standardised later on the fitting split only.
"""
import numpy as np
from ..registry import block
from .common import delta, nanquantile, slog


def bucket(x, edges):
    """0 for NaN, then 1..len(edges)+1 by np.digitize on finite values."""
    out = np.digitize(np.nan_to_num(x, nan=0.), edges) + 1
    return np.where(np.isfinite(x), out, 0)


@block('tok_action', kind='token', family='token', cardinality=4)
def tok_action(r):
    """A/D/U (0 = inconnu)."""
    return r.action


@block('tok_side', kind='token', family='token', cardinality=3)
def tok_side(r):
    return r.side


@block('tok_trade', kind='token', family='token', cardinality=3)
def tok_trade(r):
    return r.trade


@block('tok_venue', kind='token', family='token', cardinality=16)
def tok_venue(r):
    """Venue (vocabulaire du train, 0 = inconnue). Cardinalité plafonnée à 16."""
    if r.n_venue >= 16:
        raise ValueError('more than 15 venues: raise tok_venue cardinality')
    return r.venue


@block('tok_pos', kind='token', family='token', cardinality=7)
def tok_pos(r):
    """Distance au meilleur prix de son côté : inconnu, inside, best, 1, 2, 3-5, >5 ticks."""
    return bucket(r.own_dist, [-.5, .5, 1.5, 2.5, 5.5])


@block('tok_size', kind='token', family='token', cardinality=8)
def tok_size(r):
    """|flux| en lots : nan, 0, odd(<1), 1, (1,2], (2,5], (5,10], >10."""
    a = np.abs(r.flux_lots)
    b = bucket(a, [1e-9, 1 - 1e-9, 1 + 1e-9, 2 + 1e-9, 5 + 1e-9, 10 + 1e-9])
    return np.minimum(b, 7)


@block('tok_spread', kind='token', family='token', cardinality=6)
def tok_spread(r):
    """Spread en ticks : invalide, 0, 1, 2, 3-4, ≥5."""
    return bucket(r.spread_t, [.5, 1.5, 2.5, 4.5])


@block('tok_dmid', kind='token', family='token', cardinality=6)
def tok_dmid(r):
    """Δmid (demi-ticks) : invalide, ≤-1.5, baisse, 0, hausse, ≥1.5."""
    return bucket(r.dmid_h, [-1.5, -.25, .25, 1.5])


@block('tok_occ', kind='token', family='token', cardinality=5)
def tok_occ(r):
    """Occurrence de l'ordre dans la fenêtre : 1re, 2e, 3e, 4e+ (index 1..4)."""
    return np.minimum(r.occ, 3) + 1


@block('tok_prev_action', kind='token', family='token', cardinality=4)
def tok_prev_action(r):
    """Action précédente du même ordre (0 = jamais vu dans la fenêtre)."""
    return r.prev_action


@block('tok_fluxsign', kind='token', family='token', cardinality=4)
def tok_fluxsign(r):
    return bucket(r.flux, [-1e-12, 1e-12])


@block('ev_relative', kind='event', family='event')
def ev_relative(r):
    """Canaux relatifs à la fenêtre (invariants d'échelle)."""
    def scale(x):
        m = nanquantile(np.where(x > 0, x, np.nan), .5)[:, None]
        return np.where(np.isfinite(m), m, 1.)
    d, f = scale(r.depth), scale(np.abs(r.flux))
    return {'imbalance': r.imbalance, 'depth_rel': np.log1p(r.depth / d),
            'depth_change_rel': np.arcsinh(delta(r.depth) / d), 'flux_rel': slog(r.flux / f),
            'gap_log': np.log1p(r.gap)}


@block('ev_levels', kind='event', family='event_level')
def ev_levels(r):
    """Canaux en unités naturelles (niveaux) : tailles et flux en lots, spread et distance en ticks."""
    return {'log_bq_lots': np.log1p(np.where(r.size_ok, r.bq, np.nan) / r.lot),
            'log_aq_lots': np.log1p(np.where(r.size_ok, r.aq, np.nan) / r.lot),
            'slog_flux_lots': slog(r.flux_lots), 'log_spread_t': np.log1p(r.spread_t),
            'own_dist_t': np.arcsinh(r.own_dist), 'dmid_h': np.arcsinh(r.dmid_h)}
