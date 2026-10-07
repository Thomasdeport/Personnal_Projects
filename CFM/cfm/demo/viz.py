"""Figures de v_demo : un style unique, une question par figure, enregistrement PNG systématique."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PAL = {'old': '#c94f4f', 'new': '#1f9e89', 'blue': '#2a6fdb', 'orange': '#e07b39', 'gray': '#8c8c8c',
       'purple': '#7b5ea7', 'gold': '#d4a72c'}
CYCLE = ['#2a6fdb', '#e07b39', '#1f9e89', '#c94f4f', '#7b5ea7', '#8c8c8c', '#d4a72c', '#4aa3c7', '#5c9e3a', '#b0607a']
FIG_DIR = Path('figures')


def setup(fig_dir='figures'):
    global FIG_DIR
    FIG_DIR = Path(fig_dir); FIG_DIR.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 160, 'font.size': 9, 'axes.titlesize': 10,
                                'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
                                'axes.grid': True, 'grid.alpha': .25, 'axes.prop_cycle': matplotlib.cycler(color=CYCLE)})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'{name}.png', bbox_inches='tight')
    return fig


def compare(df, cols=('valid', 'stress'), title='', name='comparison', xlabel='Accuracy'):
    """Barres horizontales : un modèle par ligne, une couleur par mesure."""
    d = df.set_index('modèle')[list(cols)]
    fig, ax = plt.subplots(figsize=(7.5, .45 * len(d) + 1.4))
    y = np.arange(len(d)); h = .8 / len(cols)
    for j, c in enumerate(cols):
        ax.barh(y + j * h - .4 + h / 2, d[c], h, label=c)
        for yi, v in zip(y, d[c]):
            ax.text(v + .003, yi + j * h - .4 + h / 2, f'{v:.3f}', va='center', fontsize=7)
    ax.set_yticks(y, d.index); ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.axvline(1 / 24, color='k', lw=.6, ls=':'); ax.legend(fontsize=7, loc='lower right')
    return save(fig, name)


def curves(histories, title='Courbes d\'apprentissage', name='curves'):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    for i, (k, h) in enumerate(histories.items()):
        c = CYCLE[i % len(CYCLE)]
        axs[0].plot(h.epoch, h.train_loss, color=c, label=k)
        axs[1].plot(h.epoch, h.valid_acc, color=c, label=k)
        axs[2].plot(h.epoch, h.stress_acc, color=c, label=k)
    for a, t in zip(axs, ['Perte d\'entraînement', 'Valid (régimes tenus à l\'écart)', 'Stress (carnets peu profonds)']):
        a.set_title(t); a.set_xlabel('Époque')
    axs[0].legend(fontsize=7)
    fig.suptitle(title, fontweight='bold')
    return save(fig, name)


def waterfall(steps, title='', name='waterfall', ylabel='Accuracy'):
    """steps : liste de (étiquette, valeur). Barre = valeur ; annotation = gain par rapport à l'étape précédente."""
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(steps)), 3.4))
    vals = [v for _, v in steps]
    for i, (l, v) in enumerate(steps):
        ax.bar(i, v, color=PAL['old'] if i == 0 else PAL['new'], alpha=.9)
        ax.text(i, v + .004, f'{v:.3f}', ha='center', fontsize=8)
        if i:
            d = v - vals[i - 1]
            ax.text(i, min(vals) * .92, f'{d * 100:+.1f}', ha='center', fontsize=8, color='w', fontweight='bold')
    ax.set_xticks(range(len(steps)), [l for l, _ in steps], fontsize=7.5)
    ax.set_ylim(min(vals) * .85, max(vals) * 1.05); ax.set_ylabel(ylabel); ax.set_title(title)
    return save(fig, name)


def confusion(y, p, labels, title='Matrice de confusion (ligne = vrai titre)', name='confusion'):
    from scipy.cluster.hierarchy import leaves_list, linkage
    k = len(labels); pred = p.argmax(1)
    m = np.zeros((k, k))
    np.add.at(m, (y, pred), 1)
    m = m / m.sum(1, keepdims=True).clip(1)
    order = leaves_list(linkage(m + m.T, 'average'))          # titres qui se confondent rapprochés
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(m[np.ix_(order, order)], cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(k), np.asarray(labels)[order], fontsize=6); ax.set_yticks(range(k), np.asarray(labels)[order], fontsize=6)
    ax.set_xlabel('Titre prédit'); ax.set_ylabel('Vrai titre'); ax.set_title(title); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.8, label='part de la ligne')
    return save(fig, name)


def recall_by_class(y, preds, labels, title='Rappel par titre', name='recall'):
    fig, ax = plt.subplots(figsize=(10, 3))
    k = len(labels); x = np.arange(k); w = .8 / len(preds)
    for j, (nm, p) in enumerate(preds.items()):
        r = [(p.argmax(1)[y == c] == c).mean() if (y == c).any() else np.nan for c in range(k)]
        ax.bar(x + j * w - .4 + w / 2, r, w, label=nm)
    ax.set_xticks(x, labels); ax.set_xlabel('Titre'); ax.set_ylabel('Rappel'); ax.set_title(title); ax.legend(fontsize=7)
    return save(fig, name)


def per_class_shares(train_df, test_df, cols, labels, title, name, test_label='test (titre prédit)'):
    """Barres empilées : pour chaque titre, parts moyennes de `cols`, train (gauche) contre test (droite)."""
    k = len(labels); x = np.arange(k)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for side, (df, off, alpha) in enumerate([(train_df, -.2, 1.), (test_df, .2, .55)]):
        bottom = np.zeros(k)
        for j, c in enumerate(cols):
            v = df.reindex(range(k))[c].fillna(0).to_numpy()
            ax.bar(x + off, v, .38, bottom=bottom, color=CYCLE[j], alpha=alpha, label=c if side == 0 else None)
            bottom += v
    ax.set_xticks(x, labels); ax.set_xlabel(f'Titre — barre pleine : train ; barre claire : {test_label}')
    ax.set_ylabel('Part des événements'); ax.set_title(title); ax.legend(fontsize=7, ncol=len(cols), loc='upper right')
    return save(fig, name)


def window_anatomy(raw_arrays, tick=.01, title='Une fenêtre réelle', name='window', n=100):
    """Meilleures limites en ticks + événements (couleur = action) + parcours d'un même ordre (pointillés)."""
    num, cat, oid = raw_arrays['num'][:n], raw_arrays['cat'][:n], raw_arrays['oid'][:n]
    t = np.arange(len(num))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.step(t, num[:, 1] / tick, where='post', color=PAL['blue'], lw=1.4, label='meilleur bid')
    ax.step(t, num[:, 2] / tick, where='post', color=PAL['orange'], lw=1.4, label='meilleur ask')
    colors = {1: PAL['new'], 2: PAL['old'], 3: PAL['purple']}
    for o in np.unique(oid):
        m = oid == o
        if m.sum() > 1:
            ax.plot(t[m], num[m, 0] / tick, color='gray', lw=.6, ls=':', zorder=1)
    for a, lab in [(1, 'A ajout'), (2, 'D suppression'), (3, 'U mise à jour')]:
        m = cat[:, 1] == a
        ax.scatter(t[m], num[m, 0] / tick, s=10 + 6 * np.sqrt(np.abs(num[m, 5]) / 100), color=colors[a],
                   edgecolor='k', linewidth=.3, label=lab, zorder=3)
    tr = cat[:, 3] == 2
    ax.scatter(t[tr], num[tr, 0] / tick, marker='x', color='k', s=14, label='lié à un trade', zorder=4)
    ax.set_xlabel('Événement'); ax.set_ylabel('Prix (ticks, recentré)'); ax.set_title(title); ax.legend(fontsize=7, ncol=3)
    return save(fig, name)


def token_strip(tok, names, labels_fn=None, title='Tokens', name='tokens', n=40):
    fig, axs = plt.subplots(len(names), 1, figsize=(11, .38 * len(names) + .8), sharex=True)
    for a, j in zip(np.atleast_1d(axs), range(len(names))):
        v = tok[:n, j]; card = int(v.max()) + 1
        a.imshow(v[None], aspect='auto', cmap=ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20))[:max(card, 2)]),
                 vmin=-.5, vmax=max(card, 2) - .5)
        a.set_yticks([0], [names[j]]); a.tick_params(length=0); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    np.atleast_1d(axs)[-1].set_xlabel('Événement')
    fig.suptitle(title, fontweight='bold')
    return save(fig, name)


def embedding_map(Z, y, labels, title='Carte des représentations (t-SNE)', name='tsne', n=3000, seed=0, extra=None):
    """t-SNE sur un échantillon ; couleur = titre. `extra` : (Z_test, nom) affiché en gris par-dessous."""
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    i = rng.choice(len(Z), min(n, len(Z)), replace=False)
    pts = [Z[i]]; tags = [np.asarray(y)[i]]
    if extra is not None:
        j = rng.choice(len(extra[0]), min(n // 2, len(extra[0])), replace=False)
        pts.append(extra[0][j]); tags.append(np.full(len(j), -1))
    E = TSNE(2, init='pca', random_state=seed, perplexity=30).fit_transform(np.concatenate(pts))
    tg = np.concatenate(tags)
    fig, ax = plt.subplots(figsize=(7, 6))
    if extra is not None:
        ax.scatter(*E[tg == -1].T, s=3, color='lightgray', label=extra[1])
    sc = ax.scatter(*E[tg >= 0].T, s=4, c=tg[tg >= 0], cmap='tab20b' if len(labels) > 20 else 'tab20')
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if extra is not None:
        ax.legend(fontsize=7, markerscale=3)
    fig.colorbar(sc, ax=ax, shrink=.7, label='titre')
    return save(fig, name)
