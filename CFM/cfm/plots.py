"""Figures that each answer one question. All are saved to <lab>/figures/."""
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import per_class_recall

matplotlib.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                            'axes.grid': True, 'grid.alpha': .25, 'font.size': 9})
PALETTE = ['#2a6fdb', '#e07b39', '#1f9e89', '#c94f4f', '#7b5ea7', '#8c8c8c', '#d4a72c', '#4aa3c7', '#5c9e3a', '#b0607a']


def _save(fig, lab_dir, name):
    d = Path(lab_dir) / 'figures'; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f'{name}.png', bbox_inches='tight')
    return fig


def screen_map(df, lab_dir, top=12):
    """Q : quelles features séparent les titres (η², fit split) sans trop dériver (KS train/test) ?"""
    fig, ax = plt.subplots(figsize=(8, 5))
    fams = sorted(df.family.unique())
    for i, f in enumerate(fams):
        s = df[df.family == f]
        ax.scatter(s.ks, s.eta2, s=18, alpha=.8, color=PALETTE[i % len(PALETTE)], label=f)
    for _, r in df.nlargest(top, 'eta2').iterrows():
        ax.annotate(r.feature, (r.ks, r.eta2), fontsize=7, xytext=(3, 2), textcoords='offset points')
    ax.set(xlabel='Drift train/test (KS)', ylabel='Séparation des titres (η², fit split)',
           title=f'Utilité vs drift — {len(df)} features')
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, lab_dir, 'screen_map')


def forward_bars(df, lab_dir, name='screen_forward'):
    """Q : chaque bloc candidat, ajouté et réentraîné, améliore-t-il la validation ? et le stress ?"""
    d = df[df.candidate != '(base)'].groupby('candidate')[['d_valid_acc', 'd_valid_lo', 'd_valid_hi', 'd_stress_acc']].mean()
    d = d.sort_values('d_valid_acc')
    fig, ax = plt.subplots(figsize=(7, .35 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.barh(y, d.d_valid_acc * 100, color=PALETTE[0], alpha=.85, label='Δ valid (IC bootstrap 95 %, optimiste)')
    ax.errorbar(d.d_valid_acc * 100, y, xerr=[(d.d_valid_acc - d.d_valid_lo) * 100, (d.d_valid_hi - d.d_valid_acc) * 100],
                fmt='none', ecolor='k', lw=1)
    ax.scatter(d.d_stress_acc * 100, y, color=PALETTE[1], zorder=3, label='Δ stress')
    ax.axvline(0, color='k', lw=.8)
    ax.set_yticks(y, d.index); ax.set_xlabel('Δ accuracy (points) vs base')
    ax.set_title('Ajout d\'un bloc, réentraîné'); ax.legend(fontsize=7)
    return _save(fig, lab_dir, name)


def adversarial_bars(df, lab_dir):
    """Q : où vit le changement de distribution ?"""
    d = df.sort_values('auc_train_vs_test')
    fig, ax = plt.subplots(figsize=(6, .32 * len(d) + 1))
    ax.barh(d['blocks'].str.slice(0, 40), d.auc_train_vs_test, color=PALETTE[3])
    ax.axvline(.5, color='k', lw=.8); ax.set_xlim(.45, 1)
    ax.set(xlabel='AUC train vs test (3-fold)', title='Séparabilité train/test par bloc')
    return _save(fig, lab_dir, 'adversarial')


def histories(lab_dir, runs):
    """Q : l'apprentissage est-il mature ? l'écart train(eval)/valid grandit-il ?"""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for i, r in enumerate(runs):
        h = pd.read_csv(Path(lab_dir) / 'runs' / r / 'history.csv')
        c = PALETTE[i % len(PALETTE)]
        axs[0].plot(h.epoch, h.train_loss, color=c, label=r)
        axs[1].plot(h.epoch, h.valid_acc, color=c, label=f'{r} valid')
        axs[1].plot(h.epoch, h.train_eval_acc, color=c, ls=':', label=f'{r} train (eval)')
        axs[2].plot(h.epoch, h.stress_acc, color=c, ls='--', label=r)
    for a, t in zip(axs, ['Loss apprentissage', 'Accuracy : valid (—) vs train eval (···)', 'Stress (diagnostic, non sélectionné)']):
        a.set_title(t); a.set_xlabel('Époque')
    axs[0].legend(fontsize=7); axs[1].legend(fontsize=6)
    return _save(fig, lab_dir, 'histories')


def recall_by_class(lab_dir, run):
    """Q : quels titres sont faciles, lesquels s'effondrent en stress ?"""
    out = {}
    for part in ['valid', 'stress']:
        z = np.load(Path(lab_dir) / 'runs' / run / f'{part}.npz')
        out[part] = per_class_recall(z['y'], z['p'])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(len(out['valid']))
    ax.bar(x - .2, out['valid'], .4, color=PALETTE[0], label='valid')
    ax.bar(x + .2, out['stress'], .4, color=PALETTE[1], label='stress')
    ax.set_xticks(x); ax.set(xlabel='Titre', ylabel='Rappel', title=f'Rappel par titre — {run}'); ax.legend()
    return _save(fig, lab_dir, f'recall_{run}')


def agreement(agree, lab_dir):
    """Q : les modèles se trompent-ils sur les mêmes fenêtres ? (faible accord = blend utile)"""
    fig, ax = plt.subplots(figsize=(1.1 * len(agree) + 2, 1.0 * len(agree) + 1.5))
    im = ax.imshow(agree.values, vmin=.5, vmax=1, cmap='Blues')
    ax.set_xticks(range(len(agree)), agree.columns, rotation=45, ha='right'); ax.set_yticks(range(len(agree)), agree.index)
    for i in range(len(agree)):
        for j in range(len(agree)):
            ax.text(j, i, f'{agree.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label='part de prédictions identiques (valid)')
    ax.set_title('Accord entre modèles')
    return _save(fig, lab_dir, 'agreement')
