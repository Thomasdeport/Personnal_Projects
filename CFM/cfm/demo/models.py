"""Modèles de démonstration. Tous prennent les mêmes entrées : forward(tok, cont, ctx).

    tok  : (B, 100, T) entiers  — tokens de chaque événement
    cont : (B, 100, C) réels    — canaux continus de chaque événement (déjà standardisés)
    ctx  : (B, D) réels         — statistiques de la fenêtre (déjà standardisées)

Le point commun des réseaux séquentiels est l'ENCODAGE D'UN ÉVÉNEMENT : on additionne un vecteur appris
par token (comme des mots), plus une projection des canaux continus, plus une position. Ce qui change
d'un modèle à l'autre, c'est la façon de lire la suite des 100 événements.
"""
import torch
from torch import nn


class EventEncoder(nn.Module):
    """Un événement → un vecteur de dimension d : Σ embeddings(tokens) + W·canaux + position."""
    def __init__(self, cards, n_cont, d, dropout=.1):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, d) for c in cards)
        self.cont = nn.Linear(n_cont, d) if n_cont else None
        self.pos = nn.Parameter(torch.randn(1, 100, d) * .02)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout))

    def forward(self, tok, cont):
        x = self.pos.expand(tok.shape[0], -1, -1)
        for j, e in enumerate(self.embs):
            x = x + e(tok[..., j])
        if self.cont is not None:
            x = x + self.cont(cont)
        return self.out(x)


class Head(nn.Module):
    """Résumé de la séquence (moyenne + maximum) ⊕ contexte → 24 titres."""
    def __init__(self, d_seq, n_ctx, n_classes=24, d_ctx=64, dropout=.15):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(n_ctx, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, d_ctx), nn.GELU()) if n_ctx else None
        d_in = 2 * d_seq + (d_ctx if n_ctx else 0)
        self.mlp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def features(self, h, ctx):
        z = torch.cat([h.mean(1), h.amax(1)], -1)          # fréquence d'un motif / sa plus forte occurrence
        if self.ctx is not None:
            z = torch.cat([z, self.ctx(ctx)], -1)
        return z

    def forward(self, h, ctx):
        return self.mlp(self.features(h, ctx))


class StatsMLP(nn.Module):
    """Aucune séquence : uniquement les statistiques de fenêtre. Sert de témoin."""
    def __init__(self, n_ctx, n_classes=24, hidden=256, dropout=.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_ctx, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def features(self, tok, cont, ctx):
        return self.net[:-1](ctx)

    def forward(self, tok, cont, ctx):
        return self.net(ctx)


class ConvBlock(nn.Module):
    def __init__(self, d, dilation, dropout=.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 5, padding=2 * dilation, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # x : (B, L, d)
        return x + self.drop(torch.nn.functional.gelu(self.conv(self.norm(x).transpose(1, 2)).transpose(1, 2)))


class TokenCNN(nn.Module):
    """LE modèle simple du papier (« SignatureLite ») : 3 convolutions dilatées (1, 2, 4) → champ de 29 événements.
    Les motifs locaux (rafales d'annulations, allers-retours bid/ask) suffisent à reconnaître beaucoup de titres."""
    def __init__(self, cards, n_cont, n_ctx, d=64, layers=3, n_classes=24):
        super().__init__()
        self.enc = EventEncoder(cards, n_cont, d)
        self.blocks = nn.Sequential(*[ConvBlock(d, 2 ** i) for i in range(layers)])
        self.head = Head(d, n_ctx, n_classes)

    def features(self, tok, cont, ctx):
        return self.head.features(self.blocks(self.enc(tok, cont)), ctx)

    def forward(self, tok, cont, ctx):
        return self.head(self.blocks(self.enc(tok, cont)), ctx)


class TinyGRU(nn.Module):
    """Lecture récurrente bidirectionnelle (proche de Reaction R2)."""
    def __init__(self, cards, n_cont, n_ctx, d=64, n_classes=24):
        super().__init__()
        self.enc = EventEncoder(cards, n_cont, d)
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.head = Head(d, n_ctx, n_classes)

    def features(self, tok, cont, ctx):
        return self.head.features(self.gru(self.enc(tok, cont))[0], ctx)

    def forward(self, tok, cont, ctx):
        return self.head(self.gru(self.enc(tok, cont))[0], ctx)


class TinyTransformer(nn.Module):
    """Deux couches d'attention : chaque événement peut regarder tous les autres."""
    def __init__(self, cards, n_cont, n_ctx, d=64, heads=4, layers=2, n_classes=24):
        super().__init__()
        self.enc = EventEncoder(cards, n_cont, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, layers)
        self.head = Head(d, n_ctx, n_classes)

    def features(self, tok, cont, ctx):
        return self.head.features(self.tr(self.enc(tok, cont)), ctx)

    def forward(self, tok, cont, ctx):
        return self.head(self.tr(self.enc(tok, cont)), ctx)


class V2Hybrid(nn.Module):
    """Reconstruction SIMPLIFIÉE de l'hybride de la V2 (record 0,507), d'après sa description :
    canaux relatifs + 4 embeddings de dimension 8 → projection 128 → positions apprises → 3 blocs Transformer
    (4 têtes) → pooling par attention ; branche de statistiques MLP → 128 ; fusion 256 → 128 → 24.
    Différences assumées avec l'original : pas de biais « même ordre », pas de loss contrastive, statistiques
    approchées par les blocs base_relative + cat_freq. Ce n'est PAS le run exact soumis."""
    def __init__(self, cards, n_cont, n_ctx, d=128, n_classes=24, dropout=.25):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(c, 8) for c in cards)
        self.proj = nn.Linear(n_cont + 8 * len(cards), d)
        self.pos = nn.Parameter(torch.randn(1, 100, d) * .02)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, 3)
        self.score = nn.Linear(d, 1)
        self.stats = nn.Sequential(nn.Linear(n_ctx, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 128), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(2 * d if n_ctx else d, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def features(self, tok, cont, ctx):
        e = torch.cat([cont] + [emb(tok[..., j]) for j, emb in enumerate(self.embs)], -1)
        h = self.tr(self.proj(e) + self.pos)
        w = self.score(h).softmax(1)
        return torch.cat([(w * h).sum(1), self.stats(ctx)], -1)

    def forward(self, tok, cont, ctx):
        return self.fuse(self.features(tok, cont, ctx))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
