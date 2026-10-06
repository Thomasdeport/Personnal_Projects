"""Window-level blocks (one row per obs_id). Each block = one hypothesis family, screened as a unit."""
import numpy as np
from ..registry import block
from .common import (A_SIDE, ADD, B_SIDE, DEL, UPD, delta, nanmean, nanquantile, share, slog)

ACTIONS = {'A': ADD, 'D': DEL, 'U': UPD}


@block('base_relative', family='relative')
def base_relative(r):
    """Baseline ≈ résumés V_init/R2 : moyenne et écart-type de features normalisées par médianes locales."""
    def scale(x):
        m = nanquantile(np.where(x > 0, x, np.nan), .5)[:, None]
        return np.where(np.isfinite(m), m, 1.)
    s, d, f = scale(r.spread_t), scale(r.depth), scale(np.abs(r.flux))
    ch = {'spread_rel': np.log1p(r.spread_t / s), 'mid_change_rel': np.arcsinh(r.dmid_h / 2 / s),
          'price_pos_rel': np.arcsinh((r.price - r.mid) / r.tick / s), 'imbalance': r.imbalance,
          'depth_rel': np.log1p(r.depth / d), 'depth_change_rel': np.arcsinh(delta(r.depth) / d),
          'flux_rel': slog(r.flux / f), 'order_seen': (r.occ > 0).astype(float),
          'same_as_prev': np.c_[np.zeros(r.n), r.oid[:, 1:] == r.oid[:, :-1]].astype(float)}
    out = {}
    for k, v in ch.items():
        out[f'{k}_mean'] = nanmean(v)
        out[f'{k}_std'] = np.sqrt(np.maximum(nanmean(v ** 2) - nanmean(v) ** 2, 0))
    return out


@block('cat_freq', family='category')
def cat_freq(r):
    """Fréquences des catégories (venue, action, side, trade), modalité inconnue incluse."""
    out = {f'venue={v}': np.mean(r.venue == v, 1) for v in range(r.n_venue + 1)}
    out.update({f'action={a}': np.mean(r.action == c, 1) for a, c in ACTIONS.items()})
    out['side=B'] = np.mean(r.side == B_SIDE, 1)
    out['trade'] = np.mean(r.is_trade, 1)
    return out


@block('cat_cross', family='category')
def cat_cross(r):
    """Croisements venue×action et action×trade×side : qui fait quoi, où."""
    out = {}
    for v in range(1, r.n_venue + 1):
        for a, c in ACTIONS.items():
            out[f'v{v}_{a}'] = np.mean((r.venue == v) & (r.action == c), 1)
        out[f'v{v}_trade'] = np.mean((r.venue == v) & r.is_trade, 1)
    for a, c in ACTIONS.items():
        for sname, s in [('A', A_SIDE), ('B', B_SIDE)]:
            out[f'{a}_{sname}_trade'] = np.mean((r.action == c) & (r.side == s) & r.is_trade, 1)
    return out


@block('levels', family='level')
def levels(r):
    """Niveaux absolus en unités naturelles : profondeur et flux en lots. Discriminants ET sensibles au régime."""
    depth_l = np.log1p(r.depth / r.lot)
    return {'log_depth_q10': nanquantile(depth_l, .1), 'log_depth_q50': nanquantile(depth_l, .5),
            'log_depth_q90': nanquantile(depth_l, .9),
            'log_bq_q50': nanquantile(np.log1p(np.where(r.size_ok, r.bq, np.nan) / r.lot), .5),
            'log_aq_q50': nanquantile(np.log1p(np.where(r.size_ok, r.aq, np.nan) / r.lot), .5),
            'log_absflux_q50': nanquantile(np.log1p(np.abs(r.flux_lots)), .5),
            'log_absflux_mean': nanmean(np.log1p(np.abs(r.flux_lots)))}


@block('ticks', family='ticks')
def ticks(r):
    """Régime de tick : spread en ticks, mouvements du mid en demi-ticks, prix hors grille."""
    s, h = r.spread_t, r.dmid_h
    ok = np.isfinite(s)
    moved = np.isfinite(h) & (np.abs(h) > .25)
    out = {'spread_0': share(np.isclose(s, 0), ok), 'spread_1': share(np.isclose(s, 1), ok),
           'spread_2': share(np.isclose(s, 2), ok), 'spread_3_4': share((s > 2.5) & (s < 4.5), ok),
           'spread_5p': share(s >= 4.5, ok), 'spread_t_median': nanquantile(s, .5),
           'log_spread_t_mean': nanmean(np.log1p(s)),
           'mid_move': share(moved, np.isfinite(h)),
           'move_half_tick': share(np.isclose(np.abs(h), 1), moved),
           'move_one_tick': share(np.isclose(np.abs(h), 2), moved),
           'move_gt_tick': share(np.abs(h) > 2.25, moved)}
    m = r.mid / r.tick
    out['mid_range_t'] = np.nanmax(np.where(np.isfinite(m), m, -np.inf), 1) - np.nanmin(np.where(np.isfinite(m), m, np.inf), 1)
    out['mid_range_t'] = np.where(np.isfinite(out['mid_range_t']), out['mid_range_t'], np.nan)
    p = r.price / r.tick
    out['price_off_grid'] = share(np.abs(p - np.round(p)) > 1e-3, np.isfinite(p))
    sign = np.sign(np.where(moved, h, 0))
    idx = np.where(sign != 0, np.arange(100), -1)
    last = np.maximum.accumulate(idx, axis=1)                       # last move at or before t
    prev = np.concatenate([np.full((r.n, 1), -1), last[:, :-1]], 1)  # last move strictly before t
    has = (sign != 0) & (prev >= 0)
    prev_sign = np.take_along_axis(sign, np.maximum(prev, 0), 1)
    out['mid_reversal'] = share(sign != prev_sign, has)
    return out


@block('lots', family='lots')
def lots(r):
    """Convention de lots : flux ronds / odd lots / mixtes, tailles nulles ou négatives affichées."""
    a = np.abs(r.flux_lots)
    nz = np.isfinite(a) & (a > 0)
    rnd = np.isclose(a, np.round(a)) & (a >= 1)
    bq_l, aq_l = r.bq / r.lot, r.aq / r.lot
    sz = np.isfinite(r.bq) & np.isfinite(r.aq)
    return {'flux_zero': share(a == 0, np.isfinite(a)), 'flux_round': share(rnd, nz),
            'flux_odd': share(a < 1, nz), 'flux_mixed': share(~rnd & (a > 1), nz),
            'flux_exactly_1lot': share(np.isclose(a, 1), nz),
            'flux_ge_10lots': share(a >= 10, nz),
            'book_round': share(np.isclose(bq_l, np.round(bq_l)) & np.isclose(aq_l, np.round(aq_l)), sz),
            'size_zero_at_best': share((r.bq == 0) | (r.aq == 0), sz & r.quote_ok),
            'size_negative': share((r.bq < 0) | (r.aq < 0), sz)}


@block('position', family='position')
def position(r):
    """Où arrivent les événements par rapport au meilleur prix de leur côté, en ticks."""
    d = r.own_dist
    ok = np.isfinite(d)
    zones = {'inside': d < -.5, 'best': np.abs(d) <= .5, 'd1': (d > .5) & (d <= 1.5),
             'd2': (d > 1.5) & (d <= 2.5), 'd3_5': (d > 2.5) & (d <= 5.5), 'd6p': d > 5.5}
    out = {f'pos_{k}': share(v, ok) for k, v in zones.items()}
    out['pos_unknown'] = np.mean(~ok, 1)
    for a, c in ACTIONS.items():
        valid = ok & (r.action == c)
        out[f'{a}_at_best'] = share(zones['best'], valid)
        out[f'{a}_deeper'] = share(d > .5, valid)
    out['trade_at_best'] = share(zones['best'], ok & r.is_trade)
    out['log_dist_mean'] = nanmean(np.log1p(np.clip(d, 0, None)))
    return out


@block('orders', family='orders')
def orders(r):
    """Parcours d'ordres observés DANS la fenêtre (tronqués : 1re apparition ≠ création)."""
    first = r.occ == 0
    n_orders = first.sum(1)
    out = {'orders_per_event': n_orders / 100, 'repeat_share': np.mean(r.occ > 0, 1),
           'multi_event_orders': share((r.order_count > 1), first),
           'max_events_one_order': r.order_count.max(1) / 100,
           'same_as_prev_event': np.c_[np.zeros(r.n), r.oid[:, 1:] == r.oid[:, :-1]].mean(1),
           'gap_mean': nanmean(np.where(r.occ > 0, np.log1p(r.gap), np.nan)),
           'cross_venue_repeat': share(r.prev_venue != r.venue, r.occ > 0)}
    for a, c in ACTIONS.items():
        out[f'first_seen_{a}'] = share(r.action == c, first)
    pa, act = r.prev_action, r.action
    rep = r.occ > 0
    for p, pc in ACTIONS.items():
        for a, c in ACTIONS.items():
            out[f'{p}_to_{a}'] = share((pa == pc) & (act == c), rep)
    out['D_trade_share'] = share(r.is_trade, act == DEL)
    out['U_trade_share'] = share(r.is_trade, act == UPD)
    out['added_then_deleted'] = share((r.first_action == ADD) & (act == DEL), rep)
    return out


@block('book_dyn', family='dynamics')
def book_dyn(r):
    """Dynamique du carnet agrégé : imbalance, changements de cotes, persistance des signes."""
    imb = r.imbalance
    bid_ch = np.abs(delta(r.bid)) > 1e-9
    ask_ch = np.abs(delta(r.ask)) > 1e-9
    fs = np.sign(r.flux)
    fs0, fs1 = fs[:, :-1], fs[:, 1:]
    ok = np.isfinite(fs0) & np.isfinite(fs1) & (fs0 != 0) & (fs1 != 0)
    side_switch = r.side[:, 1:] != r.side[:, :-1]
    dd = delta(r.depth)
    return {'imb_mean': nanmean(imb), 'imb_abs_mean': nanmean(np.abs(imb)),
            'imb_std': np.sqrt(np.maximum(nanmean(imb ** 2) - nanmean(imb) ** 2, 0)),
            'bid_change': bid_ch[:, 1:].mean(1), 'ask_change': ask_ch[:, 1:].mean(1),
            'flux_sign_persist': share(fs0 == fs1, ok), 'side_switch': side_switch.mean(1),
            'depth_up': share(dd > 0, np.isfinite(dd)), 'depth_down': share(dd < 0, np.isfinite(dd)),
            'flux_pos': share(r.flux > 0, np.isfinite(r.flux))}


@block('tokens_bow', family='tokens')
def tokens_bow(r):
    """Sac de mots : action × côté × zone de prix (inside/best/near/far). 24 fréquences."""
    d = r.own_dist
    zone = np.select([d < -.5, np.abs(d) <= .5, d <= 2.5, d > 2.5], [0, 1, 2, 3], -1)
    out = {}
    for a, c in ACTIONS.items():
        for sname, s in [('A', A_SIDE), ('B', B_SIDE)]:
            for z, zname in enumerate(['in', 'best', 'near', 'far']):
                out[f'{a}{sname}_{zname}'] = np.mean((r.action == c) & (r.side == s) & (zone == z), 1)
    return out


@block('transitions', family='tokens')
def transitions(r):
    """Bigrammes : (action, côté) de l'événement t-1 → t. 36 fréquences."""
    tok = (r.action - 1) * 2 + (r.side - 1)
    tok = np.where((r.action > 0) & (r.side > 0), tok, -1)
    a, b = tok[:, :-1], tok[:, 1:]
    names = [f'{x}{s}' for x in 'ADU' for s in 'AB']
    return {f'{names[i]}>{names[j]}': np.mean((a == i) & (b == j), 1) for i in range(6) for j in range(6)}
