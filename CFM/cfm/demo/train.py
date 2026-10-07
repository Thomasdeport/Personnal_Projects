"""Une boucle d'entraînement lisible (≈ 100 lignes) : standardiser, entraîner, garder la meilleure époque.

Règles du protocole, identiques partout dans v_demo :
- la standardisation est estimée sur le FIT uniquement ;
- l'époque est choisie sur VALID (fenêtres de régimes tenus à l'écart) ;
- STRESS (carnets peu profonds, comme le test) n'est qu'un diagnostic ;
- les précisions « équilibrées » appliquent la correction de Sinkhorn (voir V5).
"""
import copy
import math
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from ..blend import sinkhorn_balance


class Normalizer:
    """Moyenne / écart-type par canal, estimés sur les fenêtres du fit ; NaN → 0 ; écrêtage à ±8."""
    def __init__(self, data, idx, sample=20000, seed=0):
        idx = np.random.default_rng(seed).permutation(idx)[:sample]
        c = data['cont'][idx].reshape(-1, data['cont'].shape[-1]).astype(np.float64)
        self.cm, self.cs = np.nanmean(c, 0), np.nanstd(c, 0)
        x = data['ctx'][idx].astype(np.float64)
        self.xm, self.xs = np.nanmean(x, 0), np.nanstd(x, 0)
        for a in ['cm', 'xm']:
            setattr(self, a, np.nan_to_num(getattr(self, a)).astype(np.float32))
        for a in ['cs', 'xs']:
            v = getattr(self, a)
            setattr(self, a, np.where(np.isfinite(v) & (v > 1e-6), v, 1.).astype(np.float32))

    def __call__(self, cont, ctx):
        f = lambda x, m, s: torch.nan_to_num((x - m) / s, nan=0., posinf=0., neginf=0.).clamp(-8, 8)
        return f(cont, torch.as_tensor(self.cm, device=cont.device), torch.as_tensor(self.cs, device=cont.device)), \
            f(ctx, torch.as_tensor(self.xm, device=ctx.device), torch.as_tensor(self.xs, device=ctx.device))


class OnDevice:
    """Toutes les fenêtres sur le GPU une fois pour toutes (≈ 1 Go) : les lots se forment sans DataLoader."""
    def __init__(self, data, device):
        self.tok = torch.as_tensor(data['tok'].astype(np.int16), device=device)
        self.cont = torch.as_tensor(data['cont'], device=device)
        self.ctx = torch.as_tensor(data['ctx'], device=device)

    def batch(self, idx):
        i = torch.as_tensor(idx, device=self.tok.device)
        return self.tok[i].long(), self.cont[i], self.ctx[i]


def depth_jitter(cont, cols, sigma):
    """Augmentation de profondeur : multiplie les tailles du carnet d'une fenêtre par γ = exp(σ·ε).
    Les canaux sont en log(1 + taille/lot) BRUTS (avant standardisation)."""
    if not sigma or not cols:
        return cont
    g = torch.exp(sigma * torch.randn(cont.shape[0], 1, 1, device=cont.device))
    cont = cont.clone()
    cont[..., cols] = torch.log1p(torch.expm1(cont[..., cols].clamp(min=0)) * g)
    return cont


@torch.no_grad()
def predict(model, dev, norm, idx, bs=2048, features=False):
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        tok, cont, ctx = dev.batch(idx[s:s + bs])
        cont, ctx = norm(cont, ctx)
        y = model.features(tok, cont, ctx) if features else model(tok, cont, ctx).softmax(-1)
        out.append(y.float().cpu().numpy())
    return np.concatenate(out)


from .core import accuracy  # noqa: E402  (même définition partout)


def train(model, lab, data, epochs=20, lr=2e-3, bs=512, wd=1e-2, depth_sigma=0., seed=0, label_smoothing=.02,
          verbose=True, max_minutes=20):
    """Entraîne `model` sur lab.split['fit'] ; renvoie (modèle à la meilleure époque, historique, normalizer, données GPU)."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = OnDevice(data, lab.device)
    norm = Normalizer(data, lab.split['fit'], seed=seed)
    cols = [i for i, n in enumerate(data['cont_names']) if n in ('log_bq_lots', 'log_aq_lots')]
    model = model.to(lab.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    fit, y = lab.split['fit'], torch.as_tensor(lab.y, device=lab.device)
    steps = epochs * math.ceil(len(fit) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    rng = np.random.default_rng(seed)
    hist, best, t0 = [], (-1, None), time.time()
    for ep in range(1, epochs + 1):
        model.train(); te = time.time(); tot = 0.
        order = rng.permutation(fit)
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            tok, cont, ctx = dev.batch(b)
            cont, ctx = norm(depth_jitter(cont, cols, depth_sigma), ctx)
            loss = lossf(model(tok, cont, ctx), y[torch.as_tensor(b, device=lab.device)])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); sched.step()
            tot += loss.item() * len(b)
        row = {'epoch': ep, 'train_loss': tot / len(order), 'seconds': round(time.time() - te, 1)}
        for part in ['valid', 'stress']:
            idx = lab.split[part]
            row[f'{part}_acc'] = accuracy(lab.y[idx], predict(model, dev, norm, idx))
        hist.append(row)
        if row['valid_acc'] > best[0]:
            best = (row['valid_acc'], copy.deepcopy(model.state_dict()))
        if verbose:
            print(f"  époque {ep:2d} | perte {row['train_loss']:.3f} | valid {row['valid_acc']:.4f} | stress {row['stress_acc']:.4f} ({row['seconds']}s)", flush=True)
        if (time.time() - t0) / 60 > max_minutes:
            print(f'  arrêt : budget de {max_minutes} min atteint'); break
    model.load_state_dict(best[1])
    return model, pd.DataFrame(hist), norm, dev


def evaluate(model, lab, data, norm, dev, name, seconds=None):
    """Précisions brutes et équilibrées sur valid et stress + probabilités (pour ensembles et figures)."""
    out = {'modèle': name, 'paramètres': sum(p.numel() for p in model.parameters())}
    probs = {}
    for part in ['valid', 'stress']:
        idx = lab.split[part]
        p = predict(model, dev, norm, idx); probs[part] = p
        out[f'{part}'] = accuracy(lab.y[idx], p)
        out[f'{part}_équilibré'] = accuracy(lab.y[idx], p, balanced=True)
    if seconds is not None:
        out['minutes'] = round(seconds / 60, 1)
    return out, probs


def test_probs(model, lab, kind, norm):
    """Probabilités sur le test (même normalizer que l'entraînement)."""
    data = lab.sequences(kind, 'test')
    dev = OnDevice(data, lab.device)
    return predict(model, dev, norm, np.arange(len(lab.test_ids)))
