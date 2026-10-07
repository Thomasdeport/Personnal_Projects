"""Ouvrir le lab et récupérer les données sous deux formes :

- un TABLEAU (une ligne par fenêtre) : statistiques de fenêtre, pour les arbres et les MLP ;
- des SÉQUENCES (100 événements par fenêtre) : tokens + canaux continus, pour les réseaux.

Deux représentations des séquences sont proposées, pour comparer à armes égales :
    'old' : ce que faisait la première version (canaux relatifs à des médianes locales + 4 catégories)
    'new' : la représentation du papier (11 tokens en ticks/lots/parcours d'ordres + canaux en unités naturelles)
"""
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import numpy as np
from .. import config, io
from ..blend import sinkhorn_balance
from ..registry import BLOCKS, compute, window_matrix

# torch n'est PAS importé ici : les notebooks d'arbres (XGBoost/LightGBM) n'en ont pas besoin, et mélanger
# leurs runtimes OpenMP avec celui de torch dans un même processus peut planter (observé sous macOS).


def has_cuda():
    return shutil.which('nvidia-smi') is not None


def accuracy(y, p, balanced=False):
    """Part de fenêtres bien classées ; balanced=True applique d'abord l'équilibrage de Sinkhorn."""
    p = sinkhorn_balance(p) if balanced else p
    return float((p.argmax(1) == y).mean())

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAB = '/kaggle/working/lab_v3'      # partagé avec V3/V4 : le cache est réutilisé s'il existe

OLD_TOKENS = ['tok_action', 'tok_side', 'tok_trade', 'tok_venue']
NEW_TOKENS = ['tok_action', 'tok_side', 'tok_trade', 'tok_venue', 'tok_pos', 'tok_size',
              'tok_spread', 'tok_dmid', 'tok_occ', 'tok_prev_action', 'tok_fluxsign']
OLD_STATS = ['base_relative', 'cat_freq']                          # ≈ résumés de la première version
NEW_STATS = ['levels', 'ticks', 'lots', 'position', 'orders', 'cat_freq']
DEPTH_CHANNELS = ('log_bq_lots', 'log_aq_lots')                    # canaux modifiés par l'augmentation de profondeur


@dataclass
class Lab:
    cfg: dict
    dir: Path
    split: dict
    y: np.ndarray
    classes: np.ndarray
    test_ids: np.ndarray
    device: object                 # torch.device pour les notebooks de réseaux, chaîne sinon
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------- tableaux ----------
    def table(self, blocks, part='train'):
        """Matrice (n_fenêtres, n_colonnes) des blocs demandés + noms des colonnes."""
        X, names, _ = window_matrix(list(blocks), self.dir, part, self.cfg['lot'])
        return np.asarray(X, dtype=np.float32), names

    # ---------- séquences ----------
    def sequences(self, kind='new', part='train'):
        """{'tok': (n,100,T) int64, 'cont': (n,100,C) float32, 'ctx': (n,D) float32, 'cards', 'cont_names'}"""
        key = (kind, part)
        if key not in self._cache:
            tokens = OLD_TOKENS if kind == 'old' else NEW_TOKENS
            events = ['ev_old'] if kind == 'old' else ['ev_relative', 'ev_levels']
            stats = OLD_STATS if kind == 'old' else NEW_STATS
            tok = np.stack([np.asarray(compute(t, self.dir, part, self.cfg['lot'])[0]) for t in tokens], -1)
            conts, names = [], []
            for b in events:
                x, nm = compute(b, self.dir, part, self.cfg['lot'])
                conts.append(np.asarray(x)); names += nm
            ctx, _ = self.table(stats, part)
            self._cache[key] = {'tok': tok.astype(np.int64), 'cont': np.concatenate(conts, -1).astype(np.float32),
                                'ctx': ctx, 'cards': [BLOCKS[t].cardinality for t in tokens], 'cont_names': names}
        return self._cache[key]

    @property
    def parts(self):
        return {k: self.split[k] for k in ['fit', 'valid', 'stress']}


def open_lab(demo=False, lab_dir=None, files=None, n_per_class=40, torch_device=True):
    """Construit (ou réutilise) le cache, toutes les features et le split. DEMO = données synthétiques, CPU.
    torch_device=False : n'importe pas torch (notebooks d'arbres et d'exploration)."""
    from ..__main__ import prepare
    over = [f'lab_dir={lab_dir or ("demo_lab_vdemo" if demo else DEFAULT_LAB)}']
    if demo:
        over += ['device=cpu', 'split.clusters_per_class=4']
    cfg = config.load(ROOT / 'configs' / 'v3_base.json', over)
    if demo and files is None:
        from ..synthetic import create
        files = create('demo_data_vdemo', n_per_class=n_per_class, n_test_per_class=10)
    split = prepare(cfg, files)
    name = 'cuda' if has_cuda() and not demo else 'cpu'
    if torch_device:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() and not demo else 'cpu')
    else:
        device = name
    print(f'lab = {cfg["lab_dir"]} | device = {device} | fit {len(split["fit"]):,} / valid {len(split["valid"]):,} '
          f'/ stress {len(split["stress"]):,} fenêtres', flush=True)
    meta = io.meta(cfg['lab_dir'])
    return Lab(cfg, Path(cfg['lab_dir']), split, np.asarray(io.load(cfg['lab_dir'], 'train')['y']).astype(np.int64),
               np.asarray(meta['classes']), np.asarray(io.load(cfg['lab_dir'], 'test')['ids']), device)
