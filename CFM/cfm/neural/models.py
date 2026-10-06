"""SignatureNet: events as words, local motifs by convolution, order relations by biased attention.

    tokens (B,100,T) ─ sum of embeddings ─┐
    cont   (B,100,C) ─ linear ────────────┼─ + position ─ LN ─ [conv blocks] ─ [encoder] ─ pool ─┐
                                                                                                ├─ fusion ─ 24
    ctx    (B,D)     ─ MLP (block dropout) ────────────────────────────────────────────────────┘

encoder ∈ {'conv', 'gru', 'conv_gru', 'conv_rel', 'rel'}
'rel' blocks = pre-norm Transformer layers whose attention gets + β_h · [same order] for each head h.
β is learned; β = 0 recovers plain attention, so the relational bias is a testable switch.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, d, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 5, padding=2 * dilation, dilation=dilation)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(self.proj(F.gelu(y)))


class RelBlock(nn.Module):
    def __init__(self, d, heads, dropout, same_order_bias=True):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.zeros(heads)) if same_order_bias else None

    def forward(self, x, same):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        bias = None
        if self.beta is not None:
            bias = (self.beta.view(1, -1, 1, 1) * same.unsqueeze(1)).to(q.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.drop.p if self.training else 0.)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.n2(x)))


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x):
        w = self.score(x).softmax(1)
        return (w * x).sum(1)


class SignatureNet(nn.Module):
    def __init__(self, cards, n_cont, n_ctx, cfg, n_classes=24):
        super().__init__()
        d, p = cfg['d_model'], cfg['dropout']
        self.encoder_name = cfg['encoder']
        self.embs = nn.ModuleList(nn.Embedding(c, d) for c in cards)
        self.cont = nn.Linear(n_cont, d) if n_cont else None
        self.pos = nn.Parameter(torch.randn(1, 100, d) * 0.02)
        self.norm_in, self.drop_in = nn.LayerNorm(d), nn.Dropout(p)
        enc = cfg['encoder']
        n_conv = cfg['conv_layers'] if 'conv' in enc else 0
        self.convs = nn.ModuleList(ConvBlock(d, 2 ** i, p) for i in range(n_conv))
        self.rels = nn.ModuleList(RelBlock(d, cfg['heads'], p, cfg['same_order_bias'])
                                  for _ in range(cfg['rel_layers'] if 'rel' in enc else 0))
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True) if 'gru' in enc else None
        self.pool = AttnPool(d)
        self.pool_proj = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU())
        cd = cfg['context_dim'] if n_ctx else 0
        self.ctx = nn.Sequential(nn.Linear(n_ctx, 128), nn.GELU(), nn.Dropout(p), nn.Linear(128, cd), nn.GELU()) if n_ctx else None
        if cfg['fusion'] == 'linear':
            self.head = nn.Linear(d + cd, n_classes)
        else:
            self.head = nn.Sequential(nn.Linear(d + cd, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, n_classes))

    def encode(self, tokens, cont, oid):
        x = self.pos.expand(tokens.shape[0], -1, -1)
        for j, emb in enumerate(self.embs):
            x = x + emb(tokens[..., j])
        if self.cont is not None:
            x = x + self.cont(cont)
        x = self.drop_in(self.norm_in(x))
        for c in self.convs:
            x = c(x)
        if self.rels:
            same = (oid.unsqueeze(2) == oid.unsqueeze(1)).float()
            same = same - torch.eye(oid.shape[1], device=oid.device)
            for r in self.rels:
                x = r(x, same)
        if self.gru is not None:
            x, _ = self.gru(x)
        return self.pool_proj(torch.cat([self.pool(x), x.mean(1), x.amax(1)], -1))

    def forward(self, tokens, cont, ctx, oid):
        z = self.encode(tokens, cont, oid)
        if self.ctx is not None:
            z = torch.cat([z, self.ctx(ctx)], -1)
        return self.head(z)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
