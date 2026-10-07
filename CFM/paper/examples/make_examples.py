"""Worked examples for the paper, computed by the REAL pipeline code (cfm.features.*).

A hand-built 20-event motif covers the interesting cases (pre-existing order, add at best, deeper add,
trade-driven delete, odd lot, quote improvement inside the spread, add-then-cancel, cross-venue order,
large deep order, spread widening). The window passed to the code repeats the motif 5 times (L=100)
with fresh order ids, so window-level blocks are well defined.

Outputs: paper/examples/tables/*.tex, paper/figures/ex_*.png. Run: python paper/examples/make_examples.py
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cfm import io                                   # noqa: E402
from cfm.features import events, window              # noqa: E402
from cfm.features.common import Raw                  # noqa: E402
from cfm.registry import BLOCKS                      # noqa: E402

OUT_T = ROOT / 'paper' / 'examples' / 'tables'; OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F = ROOT / 'paper' / 'figures'
CTX = {'tick': 0.01, 'lot': 100., 'n_venue': 6}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 170})

# t, venue, raw order id, action, side, price, bid, ask, bid_size, ask_size, trade, flux, story
MOTIF = [
    (1, 0, 5012, 'U', 'B', 0.00, 0.00, 0.01, 300, 200, 0, -100, "ordre déjà présent avant la fenêtre (1re apparition = U), réduit d'un lot"),
    (2, 1, 7731, 'A', 'A', 0.01, 0.00, 0.01, 300, 300, 0, 100, "ajout d'un lot au meilleur ask"),
    (3, 1, 2210, 'A', 'B', -0.02, 0.00, 0.01, 300, 300, 0, 200, "ajout de 2 lots, 2 ticks derrière le meilleur bid"),
    (4, 0, 7731, 'D', 'A', 0.01, 0.00, 0.01, 300, 200, 1, -100, "l'ordre de t=2 est exécuté (suppression liée à un trade)"),
    (5, 2, 9002, 'A', 'A', 0.02, 0.00, 0.01, 300, 200, 0, 37, "odd lot (37 actions), 1 tick derrière le meilleur ask"),
    (6, 0, 5012, 'D', 'B', 0.00, 0.00, 0.01, 100, 200, 0, -200, "annulation de l'ordre pré-existant"),
    (7, 3, 1180, 'D', 'A', 0.01, 0.00, 0.02, 100, 37, 1, -200, "trade vide le meilleur ask : l'ask passe à 0,02, spread 2 ticks, le mid monte d'un demi-tick"),
    (8, 1, 2210, 'U', 'B', -0.02, 0.00, 0.02, 100, 37, 0, 100, "l'ordre profond de t=3 grossit d'un lot"),
    (9, 2, 4040, 'A', 'A', 0.01, 0.00, 0.01, 100, 100, 0, 100, "ajout DANS le spread : améliore l'ask"),
    (10, 2, 4040, 'D', 'A', 0.01, 0.00, 0.02, 100, 37, 0, -100, "le même ordre est annulé aussitôt (ajouté puis supprimé)"),
    (11, 0, 3333, 'A', 'B', 0.01, 0.01, 0.02, 150, 37, 0, 150, "ajout DANS le spread côté bid, taille mixte (1,5 lot)"),
    (12, 4, 9002, 'U', 'A', 0.02, 0.01, 0.02, 150, 100, 0, 63, "l'odd lot de t=5 est complété à un lot rond"),
    (13, 4, 3333, 'D', 'B', 0.01, 0.00, 0.02, 100, 100, 1, -150, "l'ordre de t=11 est exécuté : le bid redescend"),
    (14, 5, 8888, 'A', 'B', -0.06, 0.00, 0.02, 100, 100, 0, 1000, "gros ordre (10 lots) très loin (6 ticks)"),
    (15, 5, 8888, 'U', 'B', -0.06, 0.00, 0.02, 100, 100, 0, -500, "réduit de moitié"),
    (16, 0, 1212, 'A', 'A', 0.02, 0.00, 0.02, 100, 200, 0, 100, "ajout d'un lot au meilleur ask"),
    (17, 0, 1212, 'U', 'A', 0.02, 0.00, 0.02, 100, 300, 0, 100, "le même ordre grossit"),
    (18, 3, 1212, 'D', 'A', 0.02, 0.00, 0.03, 100, 400, 1, -300, "exécuté sur une AUTRE venue (3) : l'ask passe à 0,03, spread 3 ticks"),
    (19, 1, 7070, 'A', 'A', 0.02, 0.00, 0.02, 100, 100, 0, 100, "ajout dans le spread, l'ask revient à 0,02"),
    (20, 1, 2210, 'D', 'B', -0.02, 0.00, 0.02, 100, 100, 0, -300, "annulation de l'ordre profond de t=3"),
]


def build_window(motif, reps=5):
    rows = []
    for r in range(reps):
        for (t, v, o, a, s, p, b, ak, bq, aq, tr, f, _) in motif:
            rows.append((v, o + 100_000 * r, a, s, p, b, ak, bq, aq, tr, f))
    num = np.array([[p, b, ak, bq, aq, f] for (_, _, _, _, p, b, ak, bq, aq, _, f) in rows], float)[None]
    amap, smap = {'A': 1, 'D': 2, 'U': 3}, {'A': 1, 'B': 2}
    cat = np.array([[v + 1, amap[a], smap[s], 2 if tr else 1] for (v, _, a, s, *_r, tr, _f) in rows])[None]
    oid = io.local_order_rank(np.array([[o for (_, o, *_) in rows]], float))
    return {'num': num, 'cat': cat, 'oid': oid}


TOKENS = ['tok_action', 'tok_side', 'tok_trade', 'tok_venue', 'tok_pos', 'tok_size',
          'tok_spread', 'tok_dmid', 'tok_occ', 'tok_prev_action', 'tok_fluxsign']
LABELS = {
    'tok_action': ['?', 'A', 'D', 'U'], 'tok_side': ['?', 'ask', 'bid'], 'tok_trade': ['?', 'non', 'oui'],
    'tok_venue': ['?'] + [f'v{i}' for i in range(15)],
    'tok_pos': ['?', 'in', '=', '+1', '+2', '+3–5', '>+5'],
    'tok_size': ['?', '0', 'odd', '=1 lot', '(1,2]', '(2,5]', '(5,10]', '>10'],
    'tok_spread': ['?', '0', '1', '2', '3–4', '≥5'], 'tok_dmid': ['?', '≤−1,5', 'baisse', '0', 'hausse', '≥1,5'],
    'tok_occ': ['', '1re', '2e', '3e', '4e+'], 'tok_prev_action': ['aucune', 'A', 'D', 'U'],
    'tok_fluxsign': ['?', '<0', '0', '>0']}
SHORT = {'tok_action': 'act', 'tok_side': 'côté', 'tok_trade': 'trade', 'tok_venue': 'venue', 'tok_pos': 'pos',
         'tok_size': 'taille', 'tok_spread': 'spread', 'tok_dmid': 'Δmid', 'tok_occ': 'occ',
         'tok_prev_action': 'préc.', 'tok_fluxsign': 'signe'}


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    if isinstance(x, float):
        s = f'{x:.{nd}f}'
        s = s.rstrip('0').rstrip('.') if '.' in s else s
        s = '0' if s in ('', '-0') else s
    else:
        s = str(x)
    return s.replace('.', '{,}').replace('-', '$-$')


def main():
    arr = build_window(MOTIF)
    r = Raw(arr, CTX)
    tok = {name: np.asarray(BLOCKS[name].fn(r))[0] for name in TOKENS}
    T = len(MOTIF)

    # ---------- table 1 : raw ----------
    lines = [r'\begin{tabular}{rrrcccrrrrcrl}', r'\toprule',
             r'$t$ & venue & order\_id & act. & côté & price & bid & ask & $q^b$ & $q^a$ & trade & flux\\', r'\midrule']
    for (t, v, o, a, s, p, b, ak, bq, aq, tr, f, _) in MOTIF:
        lines.append(f'{t} & {v} & {o} & {a} & {s} & {fmt(p)} & {fmt(b)} & {fmt(ak)} & {bq} & {aq} & {"oui" if tr else "non"} & {fmt(f)}\\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_T / 'raw.tex').write_text('\n'.join(lines))

    # ---------- table 2 : derived ----------
    rho, occ, prev, gap = r.oid[0], r.occ[0], r.prev_action[0], r.gap[0]
    act = {0: '—', 1: 'A', 2: 'D', 3: 'U'}
    lines = [r'\begin{tabular}{rrrrrrrrrl}', r'\toprule',
             r'$t$ & $\rho_t$ & $n_t$ & $a^{\leftarrow}_t$ & $g_t$ & $S_t$ & $h_t$ & $D_t$ & $F_t$ & lecture\\', r'\midrule']
    for i, row in enumerate(MOTIF):
        lines.append(f'{i+1} & {rho[i]} & {occ[i]} & {act[int(prev[i])]} & {gap[i]} & {fmt(float(r.spread_t[0, i]), 0)} & '
                     f'{fmt(float(r.dmid_h[0, i]), 0)} & {fmt(float(r.own_dist[0, i]), 0)} & {fmt(float(r.flux_lots[0, i]))} & '
                     f'\\parbox[t]{{6.2cm}}{{\\raggedright\\scriptsize {row[-1]}}}\\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_T / 'derived.tex').write_text('\n'.join(lines))

    # ---------- table 3 : tokens ----------
    head = ' & '.join(SHORT[n] for n in TOKENS)
    lines = [r'\begin{tabular}{r' + 'c' * len(TOKENS) + '}', r'\toprule', r'$t$ & ' + head + r'\\', r'\midrule']
    for i in range(T):
        cells = [f'{int(tok[n][i])}\\,{{\\tiny({LABELS[n][int(tok[n][i])]})}}' for n in TOKENS]
        lines.append(f'{i+1} & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_T / 'tokens.tex').write_text('\n'.join(lines))

    # ---------- table 4 : window blocks on the 100-event window ----------
    pick = {'ticks': ['spread_1', 'spread_2', 'spread_3_4', 'mid_move', 'move_half_tick'],
            'lots': ['flux_round', 'flux_odd', 'flux_mixed', 'flux_exactly_1lot', 'flux_ge_10lots'],
            'position': ['pos_inside', 'pos_best', 'pos_d1', 'pos_d2', 'pos_d6p'],
            'orders': ['orders_per_event', 'repeat_share', 'first_seen_U', 'A_to_D', 'D_trade_share', 'cross_venue_repeat']}
    lines = [r'\begin{tabular}{lll}', r'\toprule', r'Bloc & Colonne & Valeur\\', r'\midrule']
    for blk, cols in pick.items():
        vals = getattr(window, blk)(r)
        for c in cols:
            lines.append(f'\\code{{{blk}}} & \\code{{{c.replace("_", chr(92) + "_")}}} & {fmt(float(vals[c][0]), 3)}\\\\')
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_T / 'blocks.tex').write_text('\n'.join(lines))

    # ---------- table 5 : scale example (old relative vs ticks) ----------
    def const_window(spread_ticks):
        n = 100
        num = np.zeros((1, n, 6)); num[0, :, 2] = 0.01 * spread_ticks
        num[0, :, 0] = np.where(np.arange(n) % 2, num[0, :, 2], 0.)        # alternate at best ask / best bid
        num[0, :, 3] = 300; num[0, :, 4] = 200; num[0, :, 5] = np.where(np.arange(n) % 2, 100, -100)
        cat = np.stack([np.ones(n), np.where(np.arange(n) % 2, 1, 2), np.where(np.arange(n) % 2, 1, 2), np.ones(n)], -1)[None]
        return Raw({'num': num, 'cat': cat, 'oid': io.local_order_rank(np.arange(n)[None].astype(float))}, CTX)
    w1, w5 = const_window(1), const_window(5)
    b1, b5 = window.base_relative(w1), window.base_relative(w5)
    t1, t5 = window.ticks(w1), window.ticks(w5)
    lines = [r'\begin{tabular}{llcc}', r'\toprule', r'Représentation & Colonne & spread 1 tick & spread 5 ticks\\', r'\midrule']
    for c in ['spread_rel_mean', 'spread_rel_std', 'price_pos_rel_mean']:
        lines.append(f'avant (relative) & \\code{{{c.replace("_", chr(92) + "_")}}} & {fmt(float(b1[c][0]), 3)} & {fmt(float(b5[c][0]), 3)}\\\\')
    lines.append(r'\midrule')
    for c in ['spread_1', 'spread_5p', 'spread_t_median']:
        lines.append(f'maintenant (ticks) & \\code{{{c.replace("_", chr(92) + "_")}}} & {fmt(float(t1[c][0]), 3)} & {fmt(float(t5[c][0]), 3)}\\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT_T / 'scale.tex').write_text('\n'.join(lines))

    # ---------- figure : the motif, book + events + order lifecycles ----------
    t_ax = np.arange(1, T + 1)
    num = arr['num'][0, :T]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.step(t_ax, num[:, 1] / 0.01, where='post', color='#2a6fdb', lw=1.6, label='meilleur bid')
    ax.step(t_ax, num[:, 2] / 0.01, where='post', color='#e07b39', lw=1.6, label='meilleur ask')
    colors = {'A': '#1f9e89', 'D': '#c94f4f', 'U': '#7b5ea7'}
    for i, (t, v, o, a, s, p, *_rest) in enumerate(MOTIF):
        marker = 'X' if MOTIF[i][10] else ('o' if a == 'A' else ('s' if a == 'U' else 'v'))
        ax.scatter(t, p / 0.01, s=18 + 8 * abs(MOTIF[i][11]) ** .5, color=colors[a], marker=marker, zorder=3,
                   edgecolor='k', linewidth=.4)
    for o in set(m[2] for m in MOTIF):
        pts = [(m[0], m[5] / 0.01) for m in MOTIF if m[2] == o]
        if len(pts) > 1:
            ax.plot(*zip(*pts), color='gray', lw=.8, ls=':', zorder=2)
            ax.annotate(f'ordre {o}', pts[-1], xytext=(6, 5), textcoords='offset points', fontsize=6, color='gray')
    for c, lab in [('#1f9e89', 'A ajout'), ('#7b5ea7', 'U mise à jour'), ('#c94f4f', 'D suppression')]:
        ax.scatter([], [], color=c, label=lab, edgecolor='k', linewidth=.4)
    ax.scatter([], [], marker='X', color='w', edgecolor='k', label='lié à un trade')
    ax.set(xlabel='Événement $t$', ylabel='Prix (ticks, recentré)', xticks=t_ax)
    ax.legend(fontsize=6, ncol=3, loc='lower left')
    fig.tight_layout(); fig.savefig(OUT_F / 'ex_motif.png')

    # ---------- figure : token strip ----------
    fig, axs = plt.subplots(len(TOKENS), 1, figsize=(7.2, 4.6), sharex=True)
    for a_, n in zip(axs, TOKENS):
        vals = tok[n][:T]
        card = BLOCKS[n].cardinality
        cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20))[:card])
        a_.imshow(vals[None], aspect='auto', cmap=cmap, vmin=-.5, vmax=card - .5)
        for i, v in enumerate(vals):
            a_.text(i, 0, LABELS[n][int(v)], ha='center', va='center', fontsize=4.6)
        a_.set_yticks([0], [SHORT[n]]); a_.tick_params(axis='both', length=0)
        for s_ in a_.spines.values():
            s_.set_visible(False)
    axs[-1].set_xticks(range(T), range(1, T + 1)); axs[-1].set_xlabel('Événement $t$')
    fig.suptitle('Chaque colonne = un événement = 11 tokens (calculés par cfm.features.events)\n'
                 'pos : in = « dans le spread », = = au meilleur, +k = k ticks derrière', fontsize=7)
    fig.tight_layout(); fig.savefig(OUT_F / 'ex_tokens.png')
    print('tables:', sorted(p.name for p in OUT_T.glob('*.tex')), '| figures: ex_motif.png, ex_tokens.png')


if __name__ == '__main__':
    main()
