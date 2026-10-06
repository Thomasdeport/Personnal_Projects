"""Synthetic order-book windows for SOFTWARE checks only — never a proxy for challenge performance.

Prices sit on a 0.01 grid recentred on the first bid, sizes are lots of 100 plus odd lots, order ids
repeat inside a window, the test set is shallower and has another venue mix. Same CSV schema as the challenge.
"""
from pathlib import Path
import numpy as np
import pandas as pd

TICK, LOT = 0.01, 100


def _window(rng, c, test):
    large_tick = c % 3 == 0
    p_spread = [.9, .08, .02, 0] if large_tick else [.35, .3, .2, .15]
    depth_mean = (3 + (c % 8) * 2) * (0.6 if test else 1.0)
    venue_p = np.roll(np.array([.4, .2, .15, .1, .1, .05]), c % 6)
    if test:
        venue_p = venue_p * np.array([1.2, .7, 1, 1.2, 1.2, .7]); venue_p /= venue_p.sum()
    act_p = np.array([.45, .35, .2]) if c % 2 else np.array([.4, .45, .15])
    reuse = .2 + .03 * (c % 5)
    odd = .05 + .1 * (c % 4 == 1)
    bid_t, spread = 0, rng.choice(4, p=p_spread) + 1
    seen, rows = [], []
    for t in range(100):
        if rng.random() < (.05 if large_tick else .15):
            bid_t += rng.choice([-1, 1])
        if rng.random() < .15:
            spread = rng.choice(4, p=p_spread) + 1
        bid, ask = bid_t * TICK, (bid_t + spread) * TICK
        bq = (rng.geometric(1 / depth_mean)) * LOT + (rng.integers(1, LOT) if rng.random() < odd else 0)
        aq = (rng.geometric(1 / depth_mean)) * LOT + (rng.integers(1, LOT) if rng.random() < odd else 0)
        side = rng.choice(['A', 'B'])
        action = rng.choice(['A', 'D', 'U'], p=act_p)
        if seen and rng.random() < reuse:
            oid = seen[rng.integers(len(seen))]
        else:
            oid = len(seen); seen.append(oid)
        dist = rng.choice([-1, 0, 1, 2, 4], p=[.03, .55, .2, .12, .1] if c % 4 else [.02, .75, .15, .05, .03])
        price = (ask + dist * TICK) if side == 'A' else (bid - dist * TICK)
        size = LOT * rng.integers(1, 4) if rng.random() > odd else rng.integers(1, LOT)
        flux = size if action == 'A' else (-size if action == 'D' else int(rng.choice([-1, 1])) * size)
        trade = action != 'A' and rng.random() < (.1 + .05 * (c % 3))
        rows.append((rng.choice(6, p=venue_p), oid, action, side, round(price, 2), round(bid, 2), round(ask, 2),
                     bq, aq, trade, flux))
    return rows


def create(folder, n_per_class=20, n_test_per_class=6, seed=0):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cols = ['venue', 'order_id', 'action', 'side', 'price', 'bid', 'ask', 'bid_size', 'ask_size', 'trade', 'flux']
    for split, n in [('train', n_per_class), ('test', n_test_per_class)]:
        labels = np.repeat(np.arange(24), n)
        perm = rng.permutation(len(labels))
        frames, ys = [], []
        for k, j in enumerate(perm):
            obs = (0 if split == 'train' else 500_000) + k
            f = pd.DataFrame(_window(rng, labels[j], split == 'test'), columns=cols)
            f.insert(0, 'obs_id', obs)
            frames.append(f); ys.append((obs, labels[j]))
        pd.concat(frames, ignore_index=True).to_csv(folder / f'X_{split}.csv', index=False)
        y = pd.DataFrame(ys, columns=['obs_id', 'eqt_code_cat'])
        if split == 'train':
            y.sample(frac=1, random_state=seed).to_csv(folder / 'y_train.csv', index=False)  # shuffled on purpose
        else:
            y.to_csv(folder / 'y_test_TRUTH_synthetic_only.csv', index=False)
    return {'train': folder / 'X_train.csv', 'target': folder / 'y_train.csv', 'test': folder / 'X_test.csv'}
