"""Assemble neural inputs from registry blocks. Scalers are fitted on the given indices only."""
import numpy as np
import torch
from .. import io
from ..registry import BLOCKS, compute, window_matrix


def assemble(lab_dir, ncfg, split, lot):
    raw = io.load(lab_dir, split)
    toks = [np.asarray(compute(t, lab_dir, split, lot)[0]) for t in ncfg['tokens']]
    tokens = np.stack(toks, -1) if toks else np.zeros((len(raw['ids']), 100, 0), np.int16)
    cards = [BLOCKS[t].cardinality for t in ncfg['tokens']]
    evs = [np.asarray(compute(b, lab_dir, split, lot)[0]) for b in ncfg['event_blocks']]
    cont = np.concatenate(evs, -1) if evs else np.zeros((len(raw['ids']), 100, 0), np.float32)
    if ncfg['context_blocks']:
        ctx, names, fams = window_matrix(ncfg['context_blocks'], lab_dir, split, lot)
    else:
        ctx, names, fams = np.zeros((len(raw['ids']), 0), np.float32), [], []
    level_mask = np.array([f == 'level' for f in fams], bool)
    block_of = np.array([n.split(':')[0] for n in names])
    return {'tokens': tokens, 'cards': cards, 'cont': cont, 'ctx': ctx, 'oid': np.asarray(raw['oid']),
            'ids': np.asarray(raw['ids']), 'y': np.asarray(raw['y']).astype(np.int64) if 'y' in raw else None,
            'ctx_names': names, 'ctx_level_mask': level_mask, 'ctx_block': block_of}


class Scaler:
    """Per-channel mean/std on fitting rows, NaN → 0 after standardisation, clip ±8."""
    def __init__(self, cont, ctx, idx, chunk=4096):
        s = c = 0.; s2 = 0.
        for i in range(0, len(idx), chunk):
            x = cont[idx[i:i + chunk]].reshape(-1, cont.shape[-1]).astype(np.float64)
            ok = np.isfinite(x)
            s = s + np.where(ok, x, 0).sum(0); s2 = s2 + np.where(ok, x * x, 0).sum(0); c = c + ok.sum(0)
        self.cm = np.where(c > 0, s / np.maximum(c, 1), 0)
        self.cs = np.sqrt(np.maximum(s2 / np.maximum(c, 1) - self.cm ** 2, 0))
        self.cs = np.where(self.cs > 1e-6, self.cs, 1)
        x = ctx[idx].astype(np.float64)
        self.xm = np.nan_to_num(np.nanmean(x, 0)) if x.shape[1] else np.zeros(0)
        self.xs = np.nan_to_num(np.nanstd(x, 0), nan=1.) if x.shape[1] else np.zeros(0)
        self.xs = np.where(self.xs > 1e-6, self.xs, 1)

    def state(self):
        return {k: getattr(self, k).tolist() for k in ['cm', 'cs', 'xm', 'xs']}

    def cont(self, x, chunk=8192):
        out = np.empty(x.shape, np.float32)
        m, s = self.cm.astype(np.float32), self.cs.astype(np.float32)
        for i in range(0, len(x), chunk):
            out[i:i + chunk] = np.clip(np.nan_to_num((np.asarray(x[i:i + chunk]) - m) / s, nan=0.), -8, 8)
        return out

    def ctx(self, x):
        return np.clip(np.nan_to_num((x - self.xm) / self.xs, nan=0.), -8, 8).astype(np.float32)


class Batches:
    """Holds standardised tensors (on GPU if asked) and serves index batches. No DataLoader needed."""
    def __init__(self, data, scaler, device, on_device=True):
        dev = device if on_device else torch.device('cpu')
        self.device = device
        self.tokens = torch.from_numpy(data['tokens'].astype(np.int16)).to(dev)
        self.cont = torch.from_numpy(scaler.cont(data['cont'])).to(dev)
        self.ctx = torch.from_numpy(scaler.ctx(data['ctx'])).to(dev)
        self.oid = torch.from_numpy(data['oid'].astype(np.int16)).to(dev)
        self.y = torch.from_numpy(data['y']).to(dev) if data['y'] is not None else None

    def get(self, idx):
        i = torch.as_tensor(idx, device=self.tokens.device)
        out = [self.tokens[i].long(), self.cont[i], self.ctx[i], self.oid[i].long()]
        out = [t.to(self.device, non_blocking=True) for t in out]
        y = self.y[i].to(self.device) if self.y is not None else None
        return out, y
