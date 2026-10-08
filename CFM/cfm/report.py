"""Figures de qualité rapport (V5) : un style unique, une question par figure.

Chaque figure est enregistrée en PNG (200 dpi) et en PDF (vectoriel, pour LaTeX) dans <dossier>/,
et sa légende est ajoutée à <dossier>/index.md : le dossier se recopie tel quel dans paper/figures/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED = '#1f2430', '#8c8c8c'
C = {'teacher': '#7b5ea7', 'q60': '#1f9e89', 'q40': '#2a6fdb', 'all': '#e07b39', 'old': '#c94f4f',
     'v4': '#d4a72c', 'v3': '#8c8c8c', 'ok': '#5c9e3a'}
STOCKS = (list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('Dark2').colors))[:24]
OUT = Path('figures')


def setup(out_dir):
    """Style commun + dossier de sortie (l'index des légendes est remis à zéro)."""
    global OUT
    OUT = Path(out_dir); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'index.md').write_text('# Figures V5\n\n')
    matplotlib.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 200, 'font.size': 9, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.labelcolor': INK, 'axes.edgecolor': '#c8c8c8', 'xtick.color': INK,
        'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.alpha': .22, 'legend.frameon': False, 'figure.titleweight': 'bold'})


def save(fig, name, caption):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight')
    with open(OUT / 'index.md', 'a') as f:
        f.write(f'- **{name}** — {caption}\n')
    return fig


# ---------------------------------------------------------------------------------------------------------------
# 1. Le parcours et la boucle
# ---------------------------------------------------------------------------------------------------------------
def lb_journey(rows, name='v5_lb_journey'):
    """rows : liste de (étiquette, score LB, levier, est_un_contrôle). Les contrôles sont tracés à part."""
    d = pd.DataFrame(rows, columns=['label', 'lb', 'lever', 'control'])
    main = d[~d.control].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(main))
    ax.step(x, main.lb, where='mid', color=MUTED, lw=1, alpha=.6)
    ax.plot(x, main.lb, 'o', color=C['all'], ms=7, zorder=3)
    for i, r in main.iterrows():
        ax.annotate(f'{r.lb:.4f}', (i, r.lb), xytext=(0, 9), textcoords='offset points', ha='center',
                    fontsize=8.5, fontweight='bold', color=INK)
        if i:
            ax.annotate(f'{r.lever}\n{100 * (r.lb - main.lb[i - 1]):+.1f} pt', (i, r.lb), xytext=(0, -30),
                        textcoords='offset points', ha='center', fontsize=7, color=MUTED)
    for _, r in d[d.control].iterrows():   # contrôle : placé à côté de la soumission qu'il éclaire
        j = int(np.flatnonzero(main.label == r.lever)[0])
        ax.plot(j + .28, r.lb, 'x', color=C['old'], ms=8, mew=2, zorder=3)
        ax.annotate(f'{r.label}\n{r.lb:.4f}', (j + .28, r.lb), xytext=(8, -4), textcoords='offset points',
                    fontsize=7, color=C['old'])
    ax.set_xticks(x, main.label, rotation=0, fontsize=8)
    ax.set_ylabel('Précision publique (LB)'); ax.set_xlim(-.5, len(main) - .2)
    ax.set_ylim(main.lb.min() - .045, main.lb.max() + .02)
    ax.set_title(f'Du record historique à {main.label.iloc[-1]} : chaque soumission répond à une question')
    return save(fig, name, "Progression du score public, soumission par soumission, avec le levier ajouté à chaque "
                           "étape. Croix rouges : soumissions de contrôle (une seule différence).")


def self_training_diagram(name='v5_self_training_loop'):
    """Schéma de la boucle professeur → pseudo-labels → élèves → nouveau professeur."""
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.set_xlim(0, 11); ax.set_ylim(-.25, 4.3); ax.axis('off')

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                                    fc=matplotlib.colors.to_rgba(color, .1), ec=color, lw=1.4))
        ax.text(x + w / 2, y + h - .22, title, ha='center', va='top', fontsize=9, fontweight='bold', color=color)
        ax.text(x + w / 2, y + h - .55, body, ha='center', va='top', fontsize=7.4, color=INK, linespacing=1.35)

    def arrow(a, b, text='', rad=0.0, dy=.12):
        ax.annotate('', b, a, arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1,
                                              connectionstyle=f'arc3,rad={rad}'))
        if text:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha='center', fontsize=7, color=MUTED)

    box(.1, 2.55, 2.1, 1.4, 'Train étiqueté', '160 800 fenêtres\n24 titres\npériode d\'entraînement', C['v3'])
    box(.1, .45, 2.1, 1.4, 'Test non étiqueté', '81 600 fenêtres\nautre période\n(20 fenêtres / titre-jour)', C['v3'])
    box(2.9, 1.25, 2.3, 1.85, 'Professeur', 'ensemble refit\n+ alignement profondeur\n+ Sinkhorn\n+ vote entre voisins',
        C['teacher'])
    box(5.9, 1.25, 2.3, 1.85, 'Filtre', 'accord brut = lissé\npuis quota équilibré :\nq·N/24 fenêtres par titre,\nles plus confiantes',
        C['all'])
    box(8.8, 1.25, 2.1, 1.85, 'Élèves', 'SignatureNet 60 ép.\ntrain + pseudo-labels\n(poids 0,5)\nscaler : train seul',
        C['q60'])
    arrow((2.2, 3.0), (2.9, 2.6), 'labels')
    arrow((2.2, 1.3), (2.9, 1.7), 'X test')
    arrow((5.2, 2.18), (5.9, 2.18), 'probas test')
    arrow((8.2, 2.18), (8.8, 2.18), 'pseudo-\nlabels', dy=.1)
    arrow((2.2, 3.6), (9.85, 3.12), 'labels', rad=-0.12, dy=.33)
    arrow((9.85, 1.25), (4.05, 1.25), rad=-0.35)
    ax.text(6.95, -.1, 'tour suivant : les élèves deviennent le professeur', ha='center', fontsize=7.5, color=MUTED, style='italic')
    ax.set_title('Auto-apprentissage transductif : adapter le modèle à la période test sans ses labels')
    return save(fig, name, "Boucle d'auto-apprentissage. Le professeur (V3 B au tour 1, V4 C au tour 2) étiquette "
                           "les fenêtres test où ses prédictions brute et lissée concordent ; un quota par titre "
                           "évite de transmettre son biais de classes ; les élèves sont entraînés sur train + ces "
                           "fenêtres, puis deviennent le professeur du tour suivant.")


# ---------------------------------------------------------------------------------------------------------------
# 2. Pseudo-labels
# ---------------------------------------------------------------------------------------------------------------
def pseudo_selection(conf, agree, selections, y_teacher, classes, name='v5_pseudo_selection'):
    """conf : confiance du professeur (max proba) ; agree : accord brut/lissé ; selections : {nom: indices}."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={'width_ratios': [1.15, 1.6, .9]})
    bins = np.linspace(0, 1, 41)
    axs[0].hist(conf, bins, color=MUTED, alpha=.45, label=f'tout le test (n={len(conf):,})')
    axs[0].hist(conf[~agree], bins, color=C['old'], alpha=.75, label=f'désaccord brut/lissé ({(~agree).mean():.0%})')
    for (k, idx), c in zip(selections.items(), [C['q60'], C['q40']]):
        axs[0].hist(conf[idx], bins, histtype='step', lw=1.8, color=c, label=f'retenues {k} ({len(idx) / len(conf):.0%})')
    axs[0].set_xlabel('Confiance du professeur (max p)'); axs[0].set_ylabel('Fenêtres')
    axs[0].set_title('a. Qui est retenu ?'); axs[0].legend(fontsize=7, loc='upper left')

    k = len(classes); x = np.arange(k)
    eligible = np.bincount(y_teacher[agree], minlength=k)
    order = np.argsort(-eligible)
    axs[1].bar(x, eligible[order], color=MUTED, alpha=.35, label='éligibles (accord)')
    for (name_, idx), c in zip(selections.items(), [C['q60'], C['q40']]):
        axs[1].bar(x, np.bincount(y_teacher[idx], minlength=k)[order], color=c, alpha=.8, width=.55 if c == C['q40'] else .8,
                   label=f'retenues {name_}')
    axs[1].set_xticks(x, classes[order], rotation=90, fontsize=6.5)
    axs[1].set_title('b. Quota équilibré par titre prédit'); axs[1].legend(fontsize=7, ncol=3, loc='upper right')
    axs[1].set_ylim(0, eligible.max() * 1.18)

    qs = np.linspace(.05, 1, 20)
    ranked = np.sort(conf[agree])[::-1]
    axs[2].plot(qs, [ranked[:max(1, int(q * len(conf)))].min() if int(q * len(conf)) <= len(ranked) else np.nan
                     for q in qs], color=C['teacher'], lw=1.8)
    for (name_, idx), c in zip(selections.items(), [C['q60'], C['q40']]):
        axs[2].axvline(len(idx) / len(conf), color=c, ls='--', lw=1.2); axs[2].text(len(idx) / len(conf), .05, name_, color=c,
                                                                                     rotation=90, ha='right', fontsize=7.5)
    axs[2].set_xlabel('Part du test pseudo-étiquetée'); axs[2].set_ylabel('Confiance minimale retenue')
    axs[2].set_title('c. Le prix du quota'); axs[2].set_ylim(0, 1.02)
    return save(fig, name, "Sélection des pseudo-labels. (a) Confiance du professeur sur tout le test, sur les "
                           "fenêtres où ses prédictions brute et lissée divergent (exclues) et sur les fenêtres "
                           "retenues. (b) Éligibles et retenues par titre : le quota coupe les titres sur-prédits. "
                           "(c) Confiance minimale atteinte quand on étend la part pseudo-étiquetée (sans quota).")


# ---------------------------------------------------------------------------------------------------------------
# 3. Entraînement
# ---------------------------------------------------------------------------------------------------------------
def learning_curves(lab_dir, groups, name='v5_learning_curves'):
    """groups : {étiquette: (runs, couleur)} ; moyenne sur les graines, bande = min/max. Runs absents ignorés."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
    cols = [('valid_acc', 'a. Précision valid (régimes tenus à l\'écart)'),
            ('valid_logloss', 'b. Log-loss valid'), ('stress_acc', 'c. Précision stress (carnets peu profonds)')]
    for (label, (runs, c)) in groups.items():
        hs = [pd.read_csv(Path(lab_dir) / 'runs' / r / 'history.csv') for r in runs
              if (Path(lab_dir) / 'runs' / r / 'history.csv').exists()]
        if not hs:
            continue
        n = min(len(h) for h in hs)
        for ax, (col, title) in zip(axs, cols):
            M = np.stack([h[col].values[:n] for h in hs]); ep = hs[0].epoch.values[:n]
            ax.plot(ep, M.mean(0), color=c, lw=1.7, label=f'{label} (×{len(hs)})')
            ax.fill_between(ep, M.min(0), M.max(0), color=c, alpha=.15, lw=0)
            ax.set_title(title); ax.set_xlabel('Époque')
    axs[0].legend(fontsize=7.5, loc='lower right')
    for ax in (axs[0], axs[2]):
        ax.set_ylim(bottom=max(ax.get_ylim()[0], .5))
    return save(fig, name, "Courbes d'apprentissage (moyenne sur les graines, bande = min/max). Attention : la "
                           "valid des élèves est en partie contaminée, leur professeur ayant été refit sur valid et stress.")


def runs_scatter(res, name='v5_runs'):
    """res : DataFrame indexé par run avec config, valid_bal, stress_bal."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cmap = {'v5_q60': C['q60'], 'v5_q40': C['q40'], 'v4_student': C['v4'], 'v3_long': C['v3']}
    for cfg_, g in res.groupby('config'):
        ax.scatter(g.valid_bal, g.stress_bal, s=55, color=cmap.get(cfg_, MUTED), label=cfg_, edgecolor='white', zorder=3)
        ax.scatter(g.valid_bal.mean(), g.stress_bal.mean(), s=160, marker='*', color=cmap.get(cfg_, MUTED), edgecolor=INK, zorder=4)
    lo = min(res.valid_bal.min(), res.stress_bal.min()) - .01; hi = max(res.valid_bal.max(), res.stress_bal.max()) + .01
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=.7, ls=':')
    ax.set_xlabel('Valid équilibrée'); ax.set_ylabel('Stress équilibré'); ax.legend(fontsize=7.5)
    ax.set_title('Chaque graine, chaque config (étoile = moyenne)')
    return save(fig, name, "Précision équilibrée de chaque modèle seul, en valid (grappes de régimes) et en stress "
                           "(carnets peu profonds). Étoiles : moyenne de la config.")


# ---------------------------------------------------------------------------------------------------------------
# 4. Ensemble, post-traitements, voisins
# ---------------------------------------------------------------------------------------------------------------
def cascade(steps, name='v5_cascade', ylabel='Précision stress équilibrée'):
    """steps : liste de (étiquette, valeur) ; barre = valeur, annotation = gain sur l'étape précédente."""
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(steps)), 3.6))
    vals = np.array([v for _, v in steps]); x = np.arange(len(steps))
    colors = [MUTED] + [C['ok'] if d >= 0 else C['old'] for d in np.diff(vals)]
    ax.bar(x, vals, color=colors, width=.62)
    for i, v in enumerate(vals):
        ax.text(i, v + .002, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
        if i:
            ax.text(i, vals.min() - .012, f'{100 * (v - vals[i - 1]):+.1f} pt', ha='center', fontsize=7.5, color=colors[i])
    ax.set_xticks(x, [s for s, _ in steps], fontsize=8); ax.set_ylim(vals.min() - .03, vals.max() + .015)
    ax.set_ylabel(ylabel); ax.set_title('Ce que chaque étape ajoute (mesure interne)')
    return save(fig, name, "Cascade des gains internes : meilleur modèle seul, moyenne de l'ensemble, équilibrage "
                           "Sinkhorn, alignement de profondeur, vote entre voisins (pool valid ∪ stress).")


def neighbour_heatmap(g, name='v5_neighbours'):
    """g : grille de T.grid. Une carte k × α par nombre d'itérations."""
    g = g[g.rep != '(none)']; base = None
    its = sorted(g.iters.unique())
    fig, axs = plt.subplots(1, len(its), figsize=(4.3 * len(its), 3.3), squeeze=False)
    vmin, vmax = g.acc_balanced.min(), g.acc_balanced.max()
    for ax, it in zip(axs[0], its):
        M = g[g.iters == it].pivot_table(index='alpha', columns='k', values='acc_balanced')
        im = ax.imshow(M.values, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        ax.set_xticks(range(M.shape[1]), M.columns); ax.set_yticks(range(M.shape[0]), M.index)
        ax.set_xlabel('k voisins (pool)'); ax.set_ylabel('α (poids des voisins)'); ax.grid(False)
        ax.set_title(f'{it} itération{"s" if it > 1 else ""}')
        for (i, j), v in np.ndenumerate(M.values):
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7,
                    color='white' if v < (vmin + vmax) / 2 else INK)
    fig.colorbar(im, ax=axs[0].tolist(), shrink=.85, label='précision équilibrée')
    fig.suptitle('Vote entre voisins : la grille de réglage', x=.02, y=1.07, ha='left', fontsize=10.5)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight')
    with open(OUT / 'index.md', 'a') as f:
        f.write(f'- **{name}** — Grille du vote entre voisins sur le pool valid ∪ stress (représentation combo). '
                f'Au test, k est multiplié par 4 : chaque titre-jour y a ~20 fenêtres contre ~5 dans le pool.\n')
    return fig


def calibration(y, probs, name='v5_calibration', bins=12):
    """probs : {étiquette: P} sur une même partition étiquetée."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    edges = np.linspace(0, 1, bins + 1)
    for (label, P), c in zip(probs.items(), [C['v3'], C['q40'], C['q60'], C['all']]):
        conf, ok = P.max(1), P.argmax(1) == y
        b = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
        m = np.array([conf[b == i].mean() if (b == i).any() else np.nan for i in range(bins)])
        a = np.array([ok[b == i].mean() if (b == i).any() else np.nan for i in range(bins)])
        ece = np.nansum([abs(a[i] - m[i]) * (b == i).mean() for i in range(bins)])
        axs[0].plot(m, a, 'o-', color=c, ms=4, label=f'{label} (ECE {ece:.3f})')
        axs[1].hist(conf, edges, histtype='step', lw=1.6, color=c, label=label)
    axs[0].plot([0, 1], [0, 1], ls=':', color=MUTED); axs[0].set_xlabel('Confiance moyenne'); axs[0].set_ylabel('Précision')
    axs[0].set_title('a. Fiabilité (stress)'); axs[0].legend(fontsize=7.5)
    axs[1].set_xlabel('Confiance (max p)'); axs[1].set_title('b. Distribution des confiances'); axs[1].legend(fontsize=7.5)
    return save(fig, name, "Calibration sur le stress : courbe de fiabilité (diagonale = parfaitement calibré, ECE = "
                           "erreur de calibration attendue) et distribution des confiances.")


def recall_by_class(y, preds, classes, name='v5_recall_by_class'):
    """preds : {étiquette: argmax} ; titres triés par rappel du premier."""
    k = len(classes); first = next(iter(preds.values()))
    rec = {l: np.array([(p[y == c] == c).mean() for c in range(k)]) for l, p in preds.items()}
    order = np.argsort(rec[next(iter(rec))])
    fig, ax = plt.subplots(figsize=(11, 3.6)); x = np.arange(k); w = .8 / len(preds)
    for j, ((l, r), c) in enumerate(zip(rec.items(), [C['v3'], C['q40'], C['q60'], C['all']])):
        ax.bar(x + (j - (len(preds) - 1) / 2) * w, r[order], w, color=c, label=f'{l} (moy. {r.mean():.3f})')
    ax.set_xticks(x, classes[order], rotation=90, fontsize=7); ax.set_ylabel('Rappel (stress)'); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, ncol=len(preds)); ax.set_title('Rappel par titre : où se trouvent les erreurs ?')
    return save(fig, name, "Rappel par titre sur le stress, titres triés du plus difficile au plus facile.")


def confusion(y, pred, classes, name='v5_confusion'):
    """Matrice normalisée par ligne, titres réordonnés par regroupement hiérarchique des confusions."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    k = len(classes); M = np.zeros((k, k))
    np.add.at(M, (y, pred), 1); M = M / M.sum(1, keepdims=True).clip(1)
    S = (M + M.T) / 2; np.fill_diagonal(S, 0)
    o = leaves_list(linkage(1 - S / max(S.max(), 1e-9), 'average'))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(np.sqrt(M[np.ix_(o, o)]), cmap='magma_r', vmin=0, vmax=1)
    ax.set_xticks(range(k), classes[o], rotation=90, fontsize=6.5); ax.set_yticks(range(k), classes[o], fontsize=6.5)
    ax.set_xlabel('Titre prédit'); ax.set_ylabel('Vrai titre'); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.8); cb.set_label('√ part de la ligne')
    ax.set_title('Confusions sur le stress (titres regroupés par confusion)')
    return save(fig, name, "Matrice de confusion sur le stress, normalisée par ligne (échelle en racine pour voir les "
                           "petites confusions). Les titres sont ordonnés par regroupement hiérarchique : les blocs "
                           "révèlent les familles de titres que le modèle confond.")


# ---------------------------------------------------------------------------------------------------------------
# 5. Le test
# ---------------------------------------------------------------------------------------------------------------
def test_shift(teacher, subs, pseudo_mask, name='v5_test_changes'):
    """teacher : probas du professeur ; subs : {soumission: probas} ; pseudo_mask : fenêtres pseudo-étiquetées."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [1.1, 1]})
    t = teacher.argmax(1); labels = list(subs); x = np.arange(len(labels)); w = .38
    a_in = [100 * (P.argmax(1)[pseudo_mask] != t[pseudo_mask]).mean() for P in subs.values()]
    a_out = [100 * (P.argmax(1)[~pseudo_mask] != t[~pseudo_mask]).mean() for P in subs.values()]
    axs[0].bar(x - w / 2, a_in, w, color=C['q40'], label='fenêtres pseudo-étiquetées')
    axs[0].bar(x + w / 2, a_out, w, color=C['all'], label='autres fenêtres')
    for i in range(len(labels)):
        axs[0].text(x[i] - w / 2, a_in[i] + .3, f'{a_in[i]:.1f}', ha='center', fontsize=7.5)
        axs[0].text(x[i] + w / 2, a_out[i] + .3, f'{a_out[i]:.1f}', ha='center', fontsize=7.5)
    axs[0].set_xticks(x, labels, fontsize=8); axs[0].set_ylabel('% de prédictions différentes du professeur')
    axs[0].set_title('a. Où les élèves contredisent-ils le professeur ?'); axs[0].legend(fontsize=7.5)
    bins = np.linspace(0, 1, 41)
    axs[1].hist(teacher.max(1), bins, histtype='step', lw=1.8, color=C['teacher'], label='professeur V4 C')
    for (l, P), c in zip(subs.items(), [C['all'], C['q60'], C['q40']]):
        axs[1].hist(P.max(1), bins, histtype='step', lw=1.4, color=c, label=l)
    axs[1].set_xlabel('Confiance au test (max p)'); axs[1].set_title('b. Confiance au test'); axs[1].legend(fontsize=7.5)
    return save(fig, name, "Changements au test par rapport au professeur. Sur les fenêtres pseudo-étiquetées, les "
                           "élèves recopient presque toujours le professeur ; le gain (ou la perte) se joue sur les autres.")


def tsne_map(Z, pred, classes, conf=None, name='v5_tsne_test', n=4000, seed=0):
    """Carte t-SNE des représentations du test, colorée par titre prédit."""
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed); i = rng.choice(len(Z), min(n, len(Z)), replace=False)
    E = TSNE(2, init='pca', perplexity=35, random_state=seed).fit_transform(Z[i])
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    s = 6 if conf is None else 2 + 10 * conf[i] ** 2
    ax.scatter(E[:, 0], E[:, 1], c=[STOCKS[c % 24] for c in pred[i]], s=s, alpha=.75, lw=0)
    for c in np.unique(pred[i]):
        m = pred[i] == c
        if m.sum() > 15:
            cx, cy = np.median(E[m], 0)
            ax.text(cx, cy, classes[c], fontsize=7, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=.15', fc='white', ec='none', alpha=.7))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title('Représentations des fenêtres test (t-SNE), couleur = titre prédit')
    return save(fig, name, f"Carte t-SNE de {len(i):,} fenêtres test (représentations des élèves, refit), colorées par "
                           "titre prédit ; taille = confiance. Des îlots nets indiquent des titres séparables ; les zones "
                           "mêlées, les confusions restantes.")
