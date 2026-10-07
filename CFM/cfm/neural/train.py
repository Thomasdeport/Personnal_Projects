"""Neural development run (fit → select epoch on valid → score valid/stress → test_dev) and refit.

Selection: best valid accuracy (tie: log-loss). Stress is logged every epoch as a diagnostic only.
Train accuracy is measured in eval mode on a fixed fit subset with the same weights as valid.
Resume: <run>/latest.pt holds weights, optimizer, AMP scaler, history and RNG states; a different
config / split / feature cache / code refuses to resume.
"""
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .. import split as splitlib
from ..config import code_digest, digest
from ..metrics import ledger, scores
from ..registry import block_key
from .data import Batches, Scaler, assemble
from .models import SignatureNet, n_params


def device_for(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('device="cuda" demandé mais CUDA indisponible : activer le GPU ou passer device="cpu".')
    if name not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    return torch.device(name)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'py': random.getstate(), 'np': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng(s):
    random.setstate(s['py']); np.random.set_state(s['np']); torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])


def lr_at(step, total, warmup, base, schedule):
    if step < warmup:
        return base * (step + 1) / warmup
    if schedule == 'constant':
        return base
    t = min(1., (step - warmup) / max(1, total - warmup))
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))


def signature(lab_dir, cfg, idx, mode):
    n = cfg['neural']
    h = hashlib.sha256((Path(lab_dir) / 'split.npz').read_bytes()).hexdigest()[:12]
    blocks = {b: block_key(b, lab_dir, cfg['lot']) for b in n['tokens'] + n['event_blocks'] + n['context_blocks']}
    return digest([n, cfg['lot'], cfg['device'], h, blocks, code_digest(Path(__file__).parents[1]), mode,
                   np.asarray(idx).sum()])


class DepthShift:
    """Multiply book sizes by exp(log_s) per window, in raw space, on standardised inputs.
    Touches only log_bq_lots / log_aq_lots (events) and the context depth quantiles."""
    DEPTH_CONT = ('log_bq_lots', 'log_aq_lots')
    DEPTH_CTX = ('levels:log_depth_q10', 'levels:log_depth_q50', 'levels:log_depth_q90',
                 'levels:log_bq_q50', 'levels:log_aq_q50')

    def __init__(self, data, scaler, device):
        ci = [i for i, n in enumerate(data['cont_names']) if n in self.DEPTH_CONT]
        xi = [i for i, n in enumerate(data['ctx_names']) if n in self.DEPTH_CTX]
        if not ci and not xi:
            raise ValueError('depth shift: no depth channel among the inputs')
        t = lambda v: torch.tensor(np.asarray(v, dtype=np.float32), device=device)
        self.ci, self.cm, self.cs = ci, t(scaler.cm[ci]), t(scaler.cs[ci])
        self.xi, self.xm, self.xs = xi, t(scaler.xm[xi]), t(scaler.xs[xi])

    @staticmethod
    def _rescale(z, mean, std, log_s):
        raw = z * std + mean                          # back to log1p(size in lots)
        out = torch.log1p(torch.expm1(raw.clamp(min=0)) * torch.exp(log_s))
        return (out - mean) / std

    def __call__(self, cont, ctx, log_s):
        """log_s: tensor (B,) of log factors."""
        if self.ci:
            cont = cont.clone()
            cont[..., self.ci] = self._rescale(cont[..., self.ci], self.cm, self.cs, log_s[:, None, None])
        if self.xi:
            ctx = ctx.clone()
            ctx[:, self.xi] = self._rescale(ctx[:, self.xi], self.xm, self.xs, log_s[:, None])
        return cont, ctx


class Augment:
    """Train-time only.
    venue_dropout     : venue token → unknown for a share of events.
    ctx_block_dropout : zero a whole context block for a sample.
    level_jitter      : Gaussian noise (in standard deviations) on level context columns.
    depth_scale σ     : per window, book sizes × exp(N(0, σ)) — applied consistently to the event channels
                        log_bq_lots / log_aq_lots and to the context depth quantiles (raw space, then
                        re-standardised). Flux and lot-shape features are left untouched (orders stay in lots)."""
    DEPTH_CONT = ('log_bq_lots', 'log_aq_lots')
    DEPTH_CTX = ('levels:log_depth_q10', 'levels:log_depth_q50', 'levels:log_depth_q90',
                 'levels:log_bq_q50', 'levels:log_aq_q50')

    def __init__(self, ncfg, data, device, scaler=None):
        a = ncfg['aug']
        self.venue_p = a.get('venue_dropout', 0.)
        self.venue_j = ncfg['tokens'].index('tok_venue') if 'tok_venue' in ncfg['tokens'] else None
        self.block_p = a.get('ctx_block_dropout', 0.)
        blocks = list(dict.fromkeys(data['ctx_block'].tolist()))
        self.block_masks = torch.tensor(np.stack([data['ctx_block'] == b for b in blocks]) if blocks else
                                        np.zeros((0, 0)), dtype=torch.float32, device=device)
        self.jitter = a.get('level_jitter', 0.)
        self.level = torch.tensor(data['ctx_level_mask'], dtype=torch.float32, device=device)
        self.depth = a.get('depth_scale', 0.)
        if self.depth:
            if scaler is None:
                raise ValueError('depth_scale needs the scaler')
            self.shift = DepthShift(data, scaler, device)
            self.ci, self.xi = self.shift.ci, self.shift.xi

    _rescale = staticmethod(DepthShift._rescale)

    def __call__(self, tokens, cont, ctx, oid):
        if self.venue_p and self.venue_j is not None:
            drop = torch.rand(tokens.shape[:2], device=tokens.device) < self.venue_p
            tokens = tokens.clone(); tokens[..., self.venue_j][drop] = 0
        if self.block_p and len(self.block_masks):
            keep = (torch.rand(ctx.shape[0], len(self.block_masks), device=ctx.device) >= self.block_p).float()
            ctx = ctx * (1 - (1 - keep) @ self.block_masks).clamp(0, 1)
        if self.jitter and self.level.sum() > 0:
            ctx = ctx + self.jitter * torch.randn(ctx.shape[0], 1, device=ctx.device) * self.level
        if self.depth:
            cont, ctx = self.shift(cont, ctx, self.depth * torch.randn(cont.shape[0], device=cont.device))
        return tokens, cont, ctx, oid


@torch.no_grad()
def predict(model, batches, idx, bs=1024, amp=False, shift=None, log_gamma=0.):
    """shift: a DepthShift; log_gamma: fixed log factor applied to every window (test-time alignment)."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        x, _ = batches.get(idx[i:i + bs])
        if shift is not None and log_gamma:
            c, z = shift(x[1], x[2], torch.full((x[1].shape[0],), float(log_gamma), device=x[1].device))
            x = [x[0], c, z, x[3]]
        with torch.autocast(batches.device.type, dtype=torch.float16, enabled=amp):
            out.append(model(*x).float().softmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.head.out_features if hasattr(model.head, 'out_features') else 24))


def _train(lab_dir, cfg, run_dir, train_idx, eval_parts, mode, stop_epoch=None, stop_steps=None, total_epochs=None):
    """Shared loop. mode='dev' selects on valid; mode='refit' trains blindly to a fixed budget."""
    n, t = cfg['neural'], cfg['neural']['train']
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    device = device_for(cfg['device'])
    seed_all(t['seed'])
    data = assemble(lab_dir, n, 'train', cfg['lot'])
    scaler = Scaler(data['cont'], data['ctx'], train_idx)
    batches = Batches(data, scaler, device, t['data_on_gpu'] and device.type == 'cuda')
    k = int(data['y'].max()) + 1
    model = SignatureNet(data['cards'], data['cont'].shape[-1], data['ctx'].shape[1], n, k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    amp = t['amp'] and device.type == 'cuda'
    gs = (torch.amp.GradScaler('cuda', enabled=amp) if hasattr(torch.amp, 'GradScaler')
          else torch.cuda.amp.GradScaler(enabled=amp))
    aug = Augment(n, data, device, scaler)
    ema_decay = t.get('ema', 0.)
    ema = copy.deepcopy(model).eval() if ema_decay else None
    if ema is not None:
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    evaluated = ema if ema is not None else model
    spe = math.ceil(len(train_idx) / t['batch_size'])
    total_epochs = total_epochs or t['epochs']
    total, warm = total_epochs * spe, int(t['warmup_epochs'] * spe)
    sig = signature(lab_dir, cfg, train_idx, f'{mode}-{stop_epoch}-{stop_steps}-{total_epochs}')
    latest = run_dir / ('latest.pt' if mode == 'dev' else 'refit_latest.pt')
    st = {'epoch': 0, 'step': 0, 'history': [], 'best': None, 'stale': 0, 'done': False}
    if latest.exists():
        s = torch.load(latest, map_location='cpu', weights_only=False)
        if s['signature'] != sig:
            raise ValueError(f'{latest} was produced by another config/split/features/code: use a new run name')
        model.load_state_dict(s['weights']); opt.load_state_dict(s['opt']); gs.load_state_dict(s['amp'])
        if ema is not None:
            ema.load_state_dict(s['ema'])
        set_rng(s['rng']); st = s['state']
        print(f'resume from epoch {st["epoch"]}', flush=True)
    y = data['y']
    probe = train_idx[np.random.default_rng(0).permutation(len(train_idx))[:5000]]
    lossf = nn.CrossEntropyLoss(label_smoothing=t['label_smoothing'])
    g = torch.Generator().manual_seed(t['seed'])
    while not st['done']:
        g.manual_seed(t['seed'] * 1000 + st['epoch'])
        order = train_idx[torch.randperm(len(train_idx), generator=g).numpy()]
        model.train(); t0 = time.time(); tot = 0.
        for i in range(0, len(order), t['batch_size']):
            if stop_steps is not None and st['step'] >= stop_steps:
                break
            x, yb = batches.get(order[i:i + t['batch_size']])
            x = aug(*x)
            for gr in opt.param_groups:
                gr['lr'] = lr_at(st['step'], total, warm, t['lr'], t['schedule'])
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = lossf(model(*x).float(), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            opt.zero_grad(set_to_none=True)
            gs.scale(loss).backward(); gs.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            gs.step(opt); gs.update()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            st['step'] += 1; tot += loss.item() * len(yb)
        st['epoch'] += 1
        row = {'epoch': st['epoch'], 'step': st['step'], 'train_loss': tot / len(order),
               'lr': opt.param_groups[0]['lr'], 'seconds': round(time.time() - t0, 1)}
        if mode == 'dev':
            row.update({f'train_eval_{k_}': v for k_, v in scores(y[probe], predict(evaluated, batches, probe, amp=amp)).items()})
            for part, idx in eval_parts.items():
                row.update({f'{part}_{k_}': v for k_, v in scores(y[idx], predict(evaluated, batches, idx, amp=amp)).items()})
            better = st['best'] is None or (row['valid_acc'], -row['valid_logloss']) > \
                (st['best']['valid_acc'], -st['best']['valid_logloss'])
            if better:
                st['best'] = dict(row); st['stale'] = 0
                torch.save(copy.deepcopy(evaluated.state_dict()), run_dir / 'best_weights.pt')
            else:
                st['stale'] += 1
            st['done'] = st['stale'] >= t['patience'] or st['epoch'] >= total_epochs
            msg = f"valid {row['valid_acc']:.4f} stress {row.get('stress_acc', float('nan')):.4f} train(eval) {row['train_eval_acc']:.4f}"
        else:
            st['done'] = (stop_epoch is not None and st['epoch'] >= stop_epoch) or \
                         (stop_steps is not None and st['step'] >= stop_steps)
            msg = ''
        st['history'].append(row)
        torch.save({'signature': sig, 'weights': model.state_dict(), 'opt': opt.state_dict(), 'amp': gs.state_dict(),
                    'ema': ema.state_dict() if ema is not None else None, 'rng': rng_state(), 'state': st}, latest)
        pd.DataFrame(st['history']).to_csv(run_dir / ('history.csv' if mode == 'dev' else 'refit_history.csv'), index=False)
        print(f"[{run_dir.name}] ep {st['epoch']:02d} loss {row['train_loss']:.3f} {msg} ({row['seconds']}s)", flush=True)
    if mode == 'dev':
        model.load_state_dict(torch.load(run_dir / 'best_weights.pt', map_location=device))
    else:
        if ema is not None:
            model.load_state_dict(ema.state_dict())
        torch.save(model.state_dict(), run_dir / 'refit_weights.pt')   # the weights used for test predictions
    return model, batches, scaler, data, st, spe


def fit(lab_dir, cfg, name, note=''):
    lab_dir = Path(lab_dir)
    run_dir = lab_dir / 'runs' / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / 'config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        raise ValueError(f'{run_dir} already holds another config: choose another run name')
    cfg_path.write_text(json.dumps(cfg, indent=2))
    sp = splitlib.load(lab_dir)
    t0 = time.time()
    model, batches, scaler, data, st, spe = _train(lab_dir, cfg, run_dir, sp['fit'],
                                                   {'valid': sp['valid'], 'stress': sp['stress']}, 'dev')
    amp = cfg['neural']['train']['amp'] and batches.device.type == 'cuda'
    y = data['y']
    best = st['best']
    result = {'name': name, 'kind': 'neural', 'encoder': cfg['neural']['encoder'], 'params': n_params(model),
              'best_epoch': best['epoch'], 'best_step': best['step'], 'epochs_run': st['epoch'],
              'steps_per_epoch': spe, 'minutes': round((time.time() - t0) / 60, 1),
              'tokens': cfg['neural']['tokens'], 'event_blocks': cfg['neural']['event_blocks'],
              'context_blocks': cfg['neural']['context_blocks'], 'seed': cfg['neural']['train']['seed']}
    for part in ['valid', 'stress']:
        p = predict(model, batches, sp[part], amp=amp)
        np.savez(run_dir / f'{part}.npz', idx=sp[part], obs_ids=data['ids'][sp[part]], y=y[sp[part]], p=p)
        result.update({f'{part}_{k}': v for k, v in scores(y[sp[part]], p).items()})
    np.savez(run_dir / 'sealed_audit.npz', idx=sp['audit'], obs_ids=data['ids'][sp['audit']],
             p=predict(model, batches, sp['audit'], amp=amp))  # scored only by cfm.audit.open_audit
    test = assemble(lab_dir, cfg['neural'], 'test', cfg['lot'])
    tb = Batches(test, scaler, batches.device, False)
    np.savez(run_dir / 'test_dev.npz', obs_ids=test['ids'], p=predict(model, tb, np.arange(len(test['ids'])), amp=amp))
    (run_dir / 'scaler.json').write_text(json.dumps(scaler.state()))
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2))
    ledger(lab_dir, {**result, 'note': note, 'config': cfg['neural']})
    print(f"[{name}] best epoch {best['epoch']} valid {result['valid_acc']:.4f} stress {result['stress_acc']:.4f} "
          f"params {result['params']:,}", flush=True)
    return result


def refit(lab_dir, cfg, name, budget='epochs'):
    """All labels (audit included). budget='epochs': same LR schedule per epoch, stop at best_epoch
    (more updates than dev since there is more data). budget='steps': dev LR schedule and step count."""
    lab_dir = Path(lab_dir)
    run_dir = lab_dir / 'runs' / name
    res = json.loads((run_dir / 'result.json').read_text())
    y_len = len(np.load(lab_dir / 'raw' / 'train_y.npy'))
    idx = np.arange(y_len)
    if budget == 'epochs':
        model, batches, scaler, _, st, _ = _train(lab_dir, cfg, run_dir, idx, {}, 'refit', stop_epoch=res['best_epoch'])
    elif budget == 'steps':
        dev_total = cfg['neural']['train']['epochs'] * res['steps_per_epoch']
        spe = math.ceil(len(idx) / cfg['neural']['train']['batch_size'])
        model, batches, scaler, _, st, _ = _train(lab_dir, cfg, run_dir, idx, {}, 'refit', stop_steps=res['best_step'],
                                                  total_epochs=max(1, round(dev_total / spe)))
    else:
        raise ValueError(budget)
    amp = cfg['neural']['train']['amp'] and batches.device.type == 'cuda'
    test = assemble(lab_dir, cfg['neural'], 'test', cfg['lot'])
    tb = Batches(test, scaler, batches.device, False)
    p = predict(model, tb, np.arange(len(test['ids'])), amp=amp)
    np.savez(run_dir / 'test_refit.npz', obs_ids=test['ids'], p=p)
    res.update({'refit_budget': budget, 'refit_steps': st['step'], 'refit_epochs': st['epoch']})
    (run_dir / 'result.json').write_text(json.dumps(res, indent=2))
    ledger(lab_dir, {'name': name, 'kind': 'refit', 'refit_budget': budget, 'refit_steps': st['step']})
    return run_dir / 'test_refit.npz'


def _load_run(lab_dir, cfg, name, weights):
    """Rebuild a trained model with the scaler of its training population."""
    lab_dir = Path(lab_dir); run_dir = lab_dir / 'runs' / name
    n = cfg['neural']
    device = device_for(cfg['device'])
    sp = splitlib.load(lab_dir)
    data = assemble(lab_dir, n, 'train', cfg['lot'])
    idx = sp['fit'] if weights == 'dev' else np.arange(len(data['y']))
    scaler = Scaler(data['cont'], data['ctx'], idx)
    model = SignatureNet(data['cards'], data['cont'].shape[-1], data['ctx'].shape[1], n, int(data['y'].max()) + 1).to(device)
    if weights == 'dev':
        state = torch.load(run_dir / 'best_weights.pt', map_location=device)
    elif weights == 'refit':
        f = run_dir / 'refit_weights.pt'
        state = torch.load(f, map_location=device) if f.exists() else \
            torch.load(run_dir / 'refit_latest.pt', map_location=device, weights_only=False)['weights']
    else:
        raise ValueError(weights)
    model.load_state_dict(state); model.eval()
    return model, data, scaler, sp, device, n['train']['amp'] and device.type == 'cuda'


def _rows(lab_dir, cfg, weights, part, data, sp):
    if part == 'test':
        src = assemble(lab_dir, cfg['neural'], 'test', cfg['lot'])
        return src, np.arange(len(src['ids']))
    if weights == 'refit':
        raise ValueError('refit weights have seen valid/stress labels: use weights="dev" on labelled partitions')
    return data, sp[part]


@torch.no_grad()
def embed(lab_dir, cfg, name, weights='dev', parts=('valid', 'stress'), bs=1024):
    """Save fused representations to <run>/emb_<weights>_<part>.npz.

    weights='dev'  : best development checkpoint, scaler fitted on fit → use on valid/stress (never trained on them).
    weights='refit': refit checkpoint, scaler fitted on all labels → use on test only."""
    model, data, scaler, sp, device, amp = _load_run(lab_dir, cfg, name, weights)
    run_dir = Path(lab_dir) / 'runs' / name
    out = {}
    for part in parts:
        src, rows = _rows(lab_dir, cfg, weights, part, data, sp)
        b = Batches(src, scaler, device, False)
        z = []
        for i in range(0, len(rows), bs):
            x, _ = b.get(rows[i:i + bs])
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                z.append(model.features(*x).float().cpu().numpy())
        z = np.concatenate(z)
        np.savez(run_dir / f'emb_{weights}_{part}.npz', obs_ids=src['ids'][rows], z=z.astype(np.float16))
        out[part] = z
    return out


def predict_shifted(lab_dir, cfg, name, weights, part, log_gammas):
    """Test-time depth alignment: predictions with book sizes × exp(g) for each g in log_gammas.
    Saved to <run>/shift_<weights>_<part>.npz (p has shape (len(log_gammas), n, K))."""
    model, data, scaler, sp, device, amp = _load_run(lab_dir, cfg, name, weights)
    src, rows = _rows(lab_dir, cfg, weights, part, data, sp)
    b = Batches(src, scaler, device, False)
    shift = DepthShift(src, scaler, device)
    P = np.stack([predict(model, b, rows, amp=amp, shift=shift, log_gamma=g) for g in log_gammas])
    np.savez(Path(lab_dir) / 'runs' / name / f'shift_{weights}_{part}.npz', obs_ids=src['ids'][rows],
             log_gammas=np.asarray(log_gammas, dtype=np.float64), p=P)
    return P
