"""Shared, label-free event quantities in natural units (ticks, lots). Every block builds on these.

Conventions (to verify with the audit, see EXPERIMENTS.md):
- side code 1 = A (ask/sell side), 2 = B (bid/buy side); action 1=A add, 2=D delete, 3=U update; trade 2 = True.
- own_dist: distance in ticks between the event price and the best price of ITS side.
  0 = at best, >0 = deeper in the book, <0 = inside the spread (improves the quote).
- Invalid quotes (non finite, ask < bid) and invalid sizes (non finite, negative) become NaN, never 0.
"""
from functools import cached_property
import numpy as np

L = 100
A_SIDE, B_SIDE = 1, 2
ADD, DEL, UPD = 1, 2, 3


def nanmean(x, axis=1):
    """Mean over finite values; NaN when a window has none (no RuntimeWarning)."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    cnt = ok.sum(axis)
    return np.where(cnt > 0, np.where(ok, x, 0).sum(axis) / np.maximum(cnt, 1), np.nan)


def share(mask, valid=None, axis=1):
    """Fraction of events where mask holds, among `valid` events."""
    valid = np.ones_like(mask, bool) if valid is None else valid
    n = valid.sum(axis)
    return np.where(n > 0, (mask & valid).sum(axis) / np.maximum(n, 1), np.nan)


def nanquantile(x, q, axis=1):
    """Row-wise quantile over finite values (linear interpolation, = np.nanquantile), NaN if none. Fast."""
    assert axis == 1
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    s = np.sort(np.where(ok, x, np.inf), axis=1)
    cnt = ok.sum(1)
    pos = q * np.maximum(cnt - 1, 0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
    rows = np.arange(len(x))
    a, b = s[rows, lo], s[rows, hi]
    val = a + (pos - lo) * np.where(hi > lo, b - a, 0)
    return np.where(cnt > 0, val, np.nan)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def delta(x):
    return np.diff(x, axis=1, prepend=x[:, :1])


class Raw:
    def __init__(self, arrays, ctx):
        num = np.asarray(arrays['num'], dtype=np.float64)
        self.price, self.bid, self.ask, self.bq, self.aq, self.flux = np.moveaxis(num, -1, 0)
        cat = np.asarray(arrays['cat'], dtype=np.int64)
        self.venue, self.action, self.side, self.trade = np.moveaxis(cat, -1, 0)
        self.oid = np.asarray(arrays['oid'], dtype=np.int64)
        self.n = len(num)
        self.tick, self.lot, self.n_venue = ctx['tick'], ctx['lot'], ctx['n_venue']
        self.is_trade = self.trade == 2

    # ---------- quotes ----------
    @cached_property
    def quote_ok(self):
        return np.isfinite(self.bid) & np.isfinite(self.ask) & (self.ask >= self.bid)

    @cached_property
    def spread_t(self):
        return np.where(self.quote_ok, (self.ask - self.bid) / self.tick, np.nan)

    @cached_property
    def mid(self):
        return np.where(self.quote_ok, (self.bid + self.ask) / 2, np.nan)

    @cached_property
    def dmid_h(self):
        """Mid change in HALF ticks vs previous event (0 for the first event)."""
        return delta(self.mid) * 2 / self.tick

    @cached_property
    def own_dist(self):
        d = np.where(self.side == B_SIDE, self.bid - self.price,
                     np.where(self.side == A_SIDE, self.price - self.ask, np.nan))
        return np.where(self.quote_ok & np.isfinite(self.price), d / self.tick, np.nan)

    # ---------- sizes ----------
    @cached_property
    def size_ok(self):
        return np.isfinite(self.bq) & np.isfinite(self.aq) & (self.bq >= 0) & (self.aq >= 0)

    @cached_property
    def depth(self):
        return np.where(self.size_ok, self.bq + self.aq, np.nan)

    @cached_property
    def imbalance(self):
        d = self.depth
        return np.where(d > 0, (self.bq - self.aq) / np.where(d > 0, d, 1), np.nan)

    @cached_property
    def flux_lots(self):
        return self.flux / self.lot

    # ---------- order trajectories inside the window (truncated: first seen ≠ created) ----------
    @cached_property
    def _orders(self):
        n = self.n
        key = (np.arange(n)[:, None] * 128 + self.oid).ravel()
        order = np.argsort(key, kind='stable')          # within one order, positions stay ascending
        k = key[order]
        idx = np.arange(len(k))
        new = np.r_[True, k[1:] != k[:-1]]
        start = np.maximum.accumulate(np.where(new, idx, 0))
        occ = np.empty(len(k), np.int64); occ[order] = idx - start
        prev = np.full(len(k), -1, np.int64); prev[order] = np.where(new, -1, np.r_[-1, order[:-1]])
        ends = np.r_[np.flatnonzero(new)[1:], len(k)]
        size = np.repeat(ends - np.flatnonzero(new), ends - np.flatnonzero(new))
        count = np.empty(len(k), np.int64); count[order] = size
        act = self.action.ravel()
        first_act = np.empty(len(k), np.int64); first_act[order] = act[order[start]]
        prev_act = np.where(prev >= 0, act[np.maximum(prev, 0)], 0)
        pos = np.tile(np.arange(L), n)
        gap = np.where(prev >= 0, pos - prev % L, 0)
        sh = (n, L)
        return {'occ': occ.reshape(sh), 'prev': prev.reshape(sh), 'count': count.reshape(sh),
                'first_action': first_act.reshape(sh), 'prev_action': prev_act.reshape(sh),
                'gap': gap.reshape(sh)}

    @property
    def occ(self):            # 0 = first time this order is seen in the window
        return self._orders['occ']

    @property
    def order_count(self):    # events of this order inside the window
        return self._orders['count']

    @property
    def first_action(self):   # action at first sighting of this order
        return self._orders['first_action']

    @property
    def prev_action(self):    # previous action of the SAME order (0 = none)
        return self._orders['prev_action']

    @property
    def gap(self):            # events since previous occurrence of the same order (0 = none)
        return self._orders['gap']

    @cached_property
    def prev_venue(self):
        p = self._orders['prev']
        return np.where(p >= 0, self.venue.ravel()[np.maximum(p, 0)].reshape(p.shape), -1)
