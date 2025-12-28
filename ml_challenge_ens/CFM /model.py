"""Neural architectures used for the CFM challenge experiments.

This module centralises several sequential models (RNN-based, hybrid CNN /
Transformer approaches, etc.) that were previously scattered across the
repository with duplicated utilities.  The goal of this rewrite is to keep the
implementations functionally identical while dramatically simplifying the file:

- imports are defined once at the top
- utility layers (DropPath, MultiScaleCNN, …) live in a single place
- models share helper routines where possible
- obvious bugs (e.g. broken super() calls, inconsistent CNN dimensions) are fixed
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

__all__ = [
    "Benchmark",
    "AdvancedRNN",
    "RegimeAwareRNN",
    "RegimeAwareFusionNet",
    "TransRNNEncoder",
    "TCFNPlus",
    "TCFNv2",
    "TCFNPlus2",
    "TCFNPlus3",
    "TCFNPlus4",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _build_continuous_features(
    num_features: int, embed_features: List[int], encode_features: List[int]
) -> List[int]:
    """Return feature indices that are neither embedded nor sinusoidally encoded."""

    masked = set(embed_features) | set(encode_features)
    return [idx for idx in range(num_features) if idx not in masked]


def _apply_time_encoding(
    x: Tensor,
    feature_idx: List[int],
    encoder: nn.Module | None,
) -> Tensor:
    if not feature_idx:
        return torch.empty(x.size(0), x.size(1), 0, device=x.device, dtype=x.dtype)
    if encoder is None:
        raise ValueError("encoder must be provided when encode_features is non-empty")
    return encoder(x[:, :, feature_idx])


class DropPath(nn.Module):
    """Stochastic depth as proposed in https://arxiv.org/abs/1603.09382."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


class LearnableTimeEncoder(nn.Module):
    """Sin / Cos encoder with learnable frequency."""

    def __init__(self, init_scale: float = 25.0) -> None:
        super().__init__()
        self.freq = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, u: Tensor) -> Tensor:
        return torch.cat([torch.sin(u / self.freq), torch.cos(u / self.freq)], dim=-1)


class MultiScaleCNN(nn.Module):
    """Conv1D stack that captures multiple receptive fields."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.conv7 = nn.Conv1d(input_dim, hidden_dim, 7, padding=3)
        self.conv15 = nn.Conv1d(input_dim, hidden_dim, 15, padding=7)
        self.conv31 = nn.Conv1d(input_dim, hidden_dim, 31, padding=15)

        self.bn = nn.BatchNorm1d(4 * hidden_dim)
        self.act = nn.GELU()

        self.depth_ff = nn.Sequential(
            nn.Conv1d(4 * hidden_dim, 4 * hidden_dim, 1),
            nn.GELU(),
            nn.BatchNorm1d(4 * hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(input_dim, 4 * hidden_dim, 1)

        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, F]
        x_t = x.transpose(1, 2)
        y = torch.cat(
            [self.conv3(x_t), self.conv7(x_t), self.conv15(x_t), self.conv31(x_t)],
            dim=1,
        )
        y = self.act(self.bn(y))
        y = self.depth_ff(y)

        s = self.se_fc(self.se_pool(y).squeeze(-1)).unsqueeze(-1)
        y = y * s

        y = self.dropout(y + self.residual(x_t))
        return y.transpose(1, 2)  # [B, T, 4*hidden_dim]


class DualDepthwiseCNN(nn.Module):
    """Depthwise + pointwise convolutions at two scales."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.dw1 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.dw2 = nn.Conv1d(dim, dim, 13, padding=6, groups=dim)
        self.pw = nn.Conv1d(dim, hidden, 1)
        self.bn = nn.BatchNorm1d(hidden)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.dw1(x.transpose(1, 2)) + self.dw2(x.transpose(1, 2))
        y = self.act(self.bn(self.pw(y)))
        return y.transpose(1, 2)


class CrossGate(nn.Module):
    """Cross-gated fusion block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g1 = nn.Linear(dim, dim)
        self.g2 = nn.Linear(dim, dim)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        g_a = torch.sigmoid(self.g1(b))
        g_b = torch.sigmoid(self.g2(a))
        return a * g_a + b * g_b


class AttentionPool(nn.Module):
    """Simple attention pooling along time."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        w = torch.softmax(self.score(x), dim=1)
        return torch.sum(w * x, dim=1)


class GatedAttentionPooling(nn.Module):
    """More expressive attention pooling with gating."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        g = torch.tanh(self.gate(x))
        scores = torch.softmax(self.score(g), dim=1)
        return torch.sum(scores * x, dim=1)


# ---------------------------------------------------------------------------
# Benchmark GRU/LSTM baseline
# ---------------------------------------------------------------------------

class Benchmark(nn.Module):
    """Baseline GRU/LSTM with categorical embeddings and optional attention."""

    def __init__(
        self,
        num_features: int,
        num_class: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        embedding_dim: int = 8,
        d_hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = "GRU",
        attention: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.num_class = num_class
        self.embed_features = embed_features
        self.num_embed_features = num_embed_features
        self.encode_features = encode_features
        self.embedding_dim = embedding_dim
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.attention = attention
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.features_not_modified = _build_continuous_features(
            num_features, embed_features, encode_features
        )

        self.embeddings = nn.Embedding(sum(num_embed_features), embedding_dim)
        offsets = np.concatenate(([0], np.cumsum(num_embed_features)[:-1]))
        self.register_buffer(
            "embeddings_offset",
            torch.tensor(offsets, dtype=torch.long),
            persistent=False,
        )

        self.input_dim = sum(
            embedding_dim if i in embed_features else 1 for i in range(num_features)
        )

        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.input_dim,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.attention_weights = (
            nn.Linear(2 * d_hidden, 1) if attention else None
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(2 * d_hidden),
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, num_class),
        )

        self.encoder = lambda u: torch.sin(torch.pi * u / 100)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        buffer = torch.zeros(batch_size, seq_len, self.input_dim, device=x.device)

        cursor = 0
        if self.encode_features:
            encoded = self.encoder(x[:, :, self.encode_features])
            width = encoded.shape[-1]
            buffer[:, :, cursor : cursor + width] = encoded
            cursor += width

        if self.embed_features:
            cat = x[:, :, self.embed_features].long()
            cat = self.embeddings(cat + self.embeddings_offset)
            cat = cat.reshape(batch_size, seq_len, -1)
            buffer[:, :, cursor : cursor + cat.shape[-1]] = cat
            cursor += cat.shape[-1]

        if cursor < buffer.shape[-1]:
            buffer[:, :, cursor:] = x[:, :, self.features_not_modified]

        rnn_out, _ = self.rnn(buffer)
        if self.attention_weights is not None:
            scores = torch.softmax(self.attention_weights(rnn_out), dim=1)
            context = torch.sum(scores * rnn_out, dim=1)
        else:
            context = torch.mean(rnn_out, dim=1)

        return self.fc(context)


# ---------------------------------------------------------------------------
# Advanced RNN with optional Transformer front-end
# ---------------------------------------------------------------------------

class AdvancedRNN(nn.Module):
    """Hybrid positional encoder + Transformer + BiRNN."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        embedding_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.01,
        rnn_type: str = "GRU",
        attention: bool = True,
        use_transformer: bool = True,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embed_features = embed_features
        self.encode_features = encode_features
        self.attention = attention
        self.use_transformer = use_transformer

        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in num_embed_features]
        )
        self.features_not_modified = _build_continuous_features(
            num_features, embed_features, encode_features
        )

        self.input_dim = (
            embedding_dim * len(embed_features)
            + len(encode_features)
            + len(self.features_not_modified)
        )

        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=nhead,
                dim_feedforward=2 * self.input_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_transformer_layers
            )
        else:
            self.transformer = None

        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.attention_layer = (
            nn.Linear(2 * hidden_dim, 1) if attention else None
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.encoder = lambda u: torch.sin(torch.pi * u / 100)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 512, self.input_dim)  # max seq len = 512
        )

    def forward(self, x: Tensor) -> Tensor:
        parts: List[Tensor] = []
        if self.encode_features:
            parts.append(self.encoder(x[:, :, self.encode_features]))
        if self.embed_features:
            embedded = [
                emb(x[:, :, idx].long())
                for idx, emb in zip(self.embed_features, self.embeddings)
            ]
            parts.append(torch.cat(embedded, dim=-1))
        if self.features_not_modified:
            parts.append(x[:, :, self.features_not_modified])

        z = torch.cat(parts, dim=-1)
        z = z + self.positional_encoding[:, : z.size(1), :]

        if self.transformer is not None:
            z = self.transformer(z)

        rnn_out, _ = self.rnn(z)
        if self.attention_layer is not None:
            scores = torch.softmax(self.attention_layer(rnn_out), dim=1)
            context = torch.sum(scores * rnn_out, dim=1)
        else:
            context = torch.mean(rnn_out, dim=1)
        return self.fc(context)


# ---------------------------------------------------------------------------
# Regime-aware RNNs
# ---------------------------------------------------------------------------

class RegimeAwareRNN(nn.Module):
    """Drop-in RNN that mixes categorical embeddings, FiLM modulations and attention."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        nhead_cat: int = 2,
        nhead_main: int = 4,
        num_transformer_layers: int = 2,
        rnn_type: str = "GRU",
        dropout: float = 0.2,
        use_transformer: bool = True,
        use_film: bool = False,
        max_len: int = 100,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embed_features = embed_features
        self.encode_features = set(encode_features)
        self.use_transformer = use_transformer
        self.use_film = use_film
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cat, embedding_dim) for num_cat in num_embed_features]
        )
        self.n_cats = len(embed_features)
        self.cat_d_model = embedding_dim

        if self.n_cats:
            if embedding_dim % nhead_cat != 0:
                raise ValueError("embedding_dim must be divisible by nhead_cat")
            cat_layer = nn.TransformerEncoderLayer(
                d_model=self.cat_d_model,
                nhead=nhead_cat,
                dim_feedforward=4 * self.cat_d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.tab_transformer = nn.TransformerEncoder(cat_layer, num_layers=1)
        else:
            self.tab_transformer = None

        self.cont_features = [i for i in range(num_features) if i not in embed_features]
        self.encoder = lambda u: torch.sin(math.pi * u / 100)
        self.cont_proj = nn.Sequential(
            nn.Linear(len(self.cont_features), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        cat_dim = self.n_cats * embedding_dim
        self.d_model = cat_dim + hidden_dim
        if self.d_model % nhead_main != 0:
            raise ValueError("d_model must be divisible by nhead_main")

        self.pre_fuse_ln = nn.LayerNorm(self.d_model)
        self.positional_encoding = nn.Parameter(
            0.01 * torch.randn(1, max_len, self.d_model)
        )

        if use_transformer:
            main_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=nhead_main,
                dim_feedforward=2 * self.d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                main_layer, num_layers=num_transformer_layers
            )
        else:
            self.transformer_encoder = None

        if use_film:
            self.film_gen = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * self.d_model),
            )
        else:
            self.film_gen = None
        self.regime_idx: Optional[List[int]] = None

        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=self.d_model,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_pool = nn.Linear(2 * hidden_dim, 1)
        self.fc = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def set_regime_idx(self, idx_list: List[int]) -> None:
        if idx_list is None:
            self.regime_idx = None
            return
        if len(idx_list) != 4:
            raise ValueError("regime_idx must contain exactly 4 indices")
        self.regime_idx = list(idx_list)

    def forward(self, x: Tensor) -> Tensor:
        B, T, F = x.shape
        parts: List[Tensor] = []

        if self.n_cats:
            cats = [
                emb(x[:, :, feat].long())
                for feat, emb in zip(self.embed_features, self.embeddings)
            ]
            cat_btke = torch.stack(cats, dim=2)  # [B, T, K, E]
            if self.tab_transformer is not None and self.n_cats > 1:
                cat_btke = cat_btke.reshape(B * T, self.n_cats, self.embedding_dim)
                cat_btke = self.tab_transformer(cat_btke)
                cat_btke = cat_btke.view(B, T, self.n_cats, self.embedding_dim)
            parts.append(cat_btke.reshape(B, T, -1))

        cont = x[:, :, self.cont_features]
        if self.encode_features:
            rel = {feat: i for i, feat in enumerate(self.cont_features)}
            enc_pos = [rel[i] for i in self.encode_features if i in rel]
            if enc_pos:
                cont = cont.clone()
                cont[:, :, enc_pos] = self.encoder(cont[:, :, enc_pos])
        parts.append(self.cont_proj(cont))

        seq = torch.cat(parts, dim=-1)
        seq = self.pre_fuse_ln(seq + self.positional_encoding[:, :T, :])

        if self.film_gen is not None and self.regime_idx is not None:
            ridx = [i for i in self.regime_idx if 0 <= i < F]
            if ridx:
                s = x[:, :, ridx]
                feats = torch.cat([s.mean(dim=1), s.std(dim=1).clamp_min(1e-6)], dim=-1)
                gamma, beta = self.film_gen(feats).chunk(2, dim=-1)
                gamma = 1.0 + 0.1 * torch.tanh(gamma)
                beta = 0.1 * torch.tanh(beta)
                seq = gamma.unsqueeze(1) * seq + beta.unsqueeze(1)

        if self.transformer_encoder is not None:
            seq = self.transformer_encoder(seq)

        rnn_out, _ = self.rnn(seq)
        scores = torch.softmax(self.attn_pool(rnn_out), dim=1)
        pooled = torch.sum(scores * rnn_out, dim=1)
        return self.fc(pooled)


class RegimeAwareFusionNet(nn.Module):
    """Hybrid TFT-inspired model with transformer, RNN and FiLM gating."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        dropout: float = 0.1,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        rnn_type: str = "GRU",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embed_features = embed_features
        self.encode_features = encode_features

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cat, embedding_dim) for num_cat in num_embed_features]
        )
        emb_dim_total = embedding_dim * len(embed_features)
        self.cont_features = _build_continuous_features(
            num_features, embed_features, encode_features
        )
        cont_dim = len(self.cont_features)

        self.encoder = LearnableTimeEncoder()
        enc_dim = 2 * len(encode_features)
        self.input_dim = emb_dim_total + cont_dim + enc_dim

        original_nhead = nhead
        while nhead > 1 and self.input_dim % nhead != 0:
            nhead -= 1
        if self.input_dim % max(nhead, 1) != 0:
            warnings.warn(
                "Falling back to nhead=1 due to incompatible input_dim", stacklevel=2
            )
            nhead = 1
        elif nhead != original_nhead:
            warnings.warn(
                f"Adjusted nhead from {original_nhead} to {nhead} for input_dim={self.input_dim}",
                stacklevel=2,
            )

        self.var_selection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=2 * hidden_dim,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(self.input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.regime_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.Sigmoid()
        )

        self.time_attention = nn.MultiheadAttention(
            embed_dim=2 * hidden_dim, num_heads=nhead, batch_first=True
        )
        self.norm_time = nn.LayerNorm(2 * hidden_dim)
        self.feature_attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        parts: List[Tensor] = []
        if self.encode_features:
            parts.append(self.encoder(x[:, :, self.encode_features]))
        if self.embed_features:
            parts.append(
                torch.cat(
                    [
                        emb(x[:, :, feat].long())
                        for feat, emb in zip(self.embed_features, self.embeddings)
                    ],
                    dim=-1,
                )
            )
        if self.cont_features:
            parts.append(x[:, :, self.cont_features])

        z = torch.cat(parts, dim=-1)
        z = z * self.var_selection(z)
        z = self.transformer(z)

        rnn_out, _ = self.rnn(z)
        gated = self.regime_gate(rnn_out) * rnn_out

        att_out, _ = self.time_attention(gated, gated, gated)
        att_out = self.norm_time(att_out + gated)

        scores = torch.softmax(self.feature_attention(att_out), dim=1)
        context = torch.sum(scores * att_out, dim=1)
        return self.fc(context)


# ---------------------------------------------------------------------------
# Transformer + RNN hybrids (TimesNet-inspired)
# ---------------------------------------------------------------------------

class TransRNNEncoder(nn.Module):
    """Stack of Transformer layers followed by a BiRNN projected back to d_model."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = "LSTM",
        rnn_hidden: Optional[int] = None,
        rnn_layers: int = 1,
        rnn_bidirectional: bool = True,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)

        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        if rnn_hidden is None:
            divisor = 2 if rnn_bidirectional else 1
            rnn_hidden = max(32, d_model // divisor)
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=rnn_bidirectional,
        )

        out_dim = rnn_hidden * (2 if rnn_bidirectional else 1)
        self.rnn_proj = nn.Linear(out_dim, d_model)
        self.regime_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.drop_path = DropPath(drop_path_rate)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        z = self.transformer(x)
        rnn_out, _ = self.rnn(z)
        rnn_proj = self.regime_gate(self.rnn_proj(rnn_out)) * self.rnn_proj(rnn_out)
        return self.norm_out(z + self.drop_path(rnn_proj))


class TCFNPlus(nn.Module):
    """Initial dual-branch CNN + Transformer fusion."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        dropout: float = 0.25,
        nhead: int = 2,
        num_transformer_layers: int = 2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embed_features = embed_features
        self.encode_features = encode_features

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cat, embedding_dim) for num_cat in num_embed_features]
        )
        emb_dim_total = embedding_dim * len(embed_features)
        self.cont_features = _build_continuous_features(
            num_features, embed_features, encode_features
        )
        cont_dim = len(self.cont_features)

        self.time_encoder = LearnableTimeEncoder()
        enc_dim = 2 * len(encode_features)
        self.input_dim = emb_dim_total + cont_dim + enc_dim

        self.var_selection = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.Sigmoid()
        )

        self.cnn_branch = MultiScaleCNN(self.input_dim, hidden_dim, dropout)
        cnn_dim = 4 * hidden_dim

        layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=2 * hidden_dim,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_branch = nn.TransformerEncoder(
            layer, num_layers=num_transformer_layers
        )

        fusion_dim = hidden_dim * 2
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.tr_proj = nn.Linear(self.input_dim, fusion_dim)
        self.fusion_attention = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.time_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=nhead, batch_first=True
        )
        self.norm_time = nn.LayerNorm(fusion_dim)
        self.regime_gate = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.fc = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        parts: List[Tensor] = []
        if self.encode_features:
            parts.append(self.time_encoder(x[:, :, self.encode_features]))
        if self.embed_features:
            parts.append(
                torch.cat(
                    [
                        emb(x[:, :, feat].long())
                        for feat, emb in zip(self.embed_features, self.embeddings)
                    ],
                    dim=-1,
                )
            )
        if self.cont_features:
            parts.append(x[:, :, self.cont_features])

        z = torch.cat(parts, dim=-1)
        z = z * self.var_selection(z)

        cnn_out = self.cnn_branch(z)
        tr_out = self.transformer_branch(z)

        cnn_proj = self.cnn_proj(cnn_out)
        tr_proj = self.tr_proj(tr_out)
        weights = self.fusion_attention((cnn_proj + tr_proj).mean(dim=1))
        fused = (
            weights[:, 0].unsqueeze(-1).unsqueeze(-1) * cnn_proj
            + weights[:, 1].unsqueeze(-1).unsqueeze(-1) * tr_proj
        )

        attn_out, _ = self.time_attention(fused, fused, fused)
        attn_out = self.norm_time(attn_out + fused)
        gated = self.regime_gate(attn_out) * attn_out
        context = gated.mean(dim=1)
        return self.fc(context)


class TCFNv2(nn.Module):
    """CNN branch in parallel with a Transformer→RNN (TransRNN) branch."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        hidden_dim: int = 128,
        embedding_dim: int = 16,
        dropout: float = 0.25,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        drop_path_rate: float = 0.1,
        rnn_type: str = "LSTM",
        rnn_layers: int = 1,
        rnn_bidirectional: bool = True,
        rnn_hidden: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embed_features = embed_features
        self.encode_features = encode_features

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cat, embedding_dim) for num_cat in num_embed_features]
        )
        emb_dim_total = embedding_dim * len(embed_features)
        self.cont_features = _build_continuous_features(
            num_features, embed_features, encode_features
        )
        cont_dim = len(self.cont_features)

        self.time_encoder = LearnableTimeEncoder()
        enc_dim = 2 * len(encode_features)
        self.input_dim = emb_dim_total + cont_dim + enc_dim

        self.feature_attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.GELU(),
            nn.Linear(self.input_dim, self.input_dim),
            nn.Softmax(dim=-1),
        )
        self.var_selection = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.Sigmoid()
        )

        self.cnn_branch = MultiScaleCNN(self.input_dim, hidden_dim, dropout)
        cnn_dim = 4 * hidden_dim

        self.transrnn = TransRNNEncoder(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=2 * hidden_dim,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            rnn_layers=rnn_layers,
            rnn_bidirectional=rnn_bidirectional,
            rnn_hidden=rnn_hidden,
            drop_path_rate=drop_path_rate,
        )

        fusion_dim = hidden_dim * 2
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.tr_proj = nn.Linear(self.input_dim, fusion_dim)
        self.drop_path = DropPath(drop_path_rate)
        self.cross_mix = nn.Linear(fusion_dim, fusion_dim)
        self.fusion_score = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

        global_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=1)
        self.pool_attn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.Tanh(),
            nn.Linear(fusion_dim // 4, 1),
        )
        self.regime_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        parts: List[Tensor] = []
        if self.encode_features:
            parts.append(self.time_encoder(x[:, :, self.encode_features]))
        if self.embed_features:
            parts.append(
                torch.cat(
                    [
                        emb(x[:, :, feat].long())
                        for feat, emb in zip(self.embed_features, self.embeddings)
                    ],
                    dim=-1,
                )
            )
        if self.cont_features:
            parts.append(x[:, :, self.cont_features])

        z = torch.cat(parts, dim=-1)
        z = z * self.feature_attention(z)
        z = z * self.var_selection(z)

        cnn_out = self.cnn_branch(z)
        tr_out = self.transrnn(z)

        cnn_latent = self.drop_path(self.cnn_proj(cnn_out))
        tr_latent = self.drop_path(self.tr_proj(tr_out))

        summed = cnn_latent + tr_latent
        cnn_latent = cnn_latent + self.cross_mix(summed - cnn_latent)
        tr_latent = tr_latent + self.cross_mix(summed - tr_latent)

        scores = torch.softmax(
            torch.cat(
                [self.fusion_score(cnn_latent.mean(dim=1)), self.fusion_score(tr_latent.mean(dim=1))],
                dim=-1,
            ),
            dim=-1,
        )
        fused = (
            scores[:, 0].unsqueeze(-1).unsqueeze(-1) * cnn_latent
            + scores[:, 1].unsqueeze(-1).unsqueeze(-1) * tr_latent
        )

        fused = self.global_encoder(fused)
        gated = self.regime_gate(fused) * fused
        attn_w = torch.softmax(self.pool_attn(gated), dim=1)
        context = torch.sum(gated * attn_w, dim=1)
        return self.fc(context)


class TCFNPlus2(nn.Module):
    """Improved TCFN+ with conv pre-processing and richer fusion."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        dropout: float = 0.25,
        nhead: int = 2,
        num_transformer_layers: int = 2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embed_features = embed_features
        self.encode_features = encode_features

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cat, embedding_dim) for num_cat in num_embed_features]
        )
        emb_dim_total = embedding_dim * len(embed_features)
        self.cont_features = _build_continuous_features(
            num_features, embed_features, encode_features
        )
        cont_dim = len(self.cont_features)

        self.time_encoder = LearnableTimeEncoder()
        enc_dim = 2 * len(encode_features)
        self.input_dim = emb_dim_total + cont_dim + enc_dim

        self.var_selection = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.Sigmoid()
        )

        self.cnn_branch = MultiScaleCNN(self.input_dim, hidden_dim, dropout)
        cnn_dim = 4 * hidden_dim

        self.pre_conv = nn.Conv1d(self.input_dim, self.input_dim, 3, padding=1)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=num_transformer_layers
        )

        fusion_dim = 2 * hidden_dim
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.tr_proj = nn.Linear(self.input_dim, fusion_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.time_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=nhead, batch_first=True
        )
        self.norm_time = nn.LayerNorm(fusion_dim)
        self.regime_gate = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
        self.drop_path = DropPath(0.1)
        self.fc = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        parts: List[Tensor] = []
        if self.encode_features:
            parts.append(self.time_encoder(x[:, :, self.encode_features]))
        if self.embed_features:
            parts.append(
                torch.cat(
                    [
                        emb(x[:, :, feat].long())
                        for feat, emb in zip(self.embed_features, self.embeddings)
                    ],
                    dim=-1,
                )
            )
        if self.cont_features:
            parts.append(x[:, :, self.cont_features])

        z = torch.cat(parts, dim=-1)
        z = z * self.var_selection(z)

        cnn_out = self.cnn_branch(z)
        tr_in = self.pre_conv(z.transpose(1, 2)).transpose(1, 2)
        tr_out = self.transformer(tr_in)

        c = self.cnn_proj(cnn_out)
        t = self.tr_proj(tr_out)

        gate = self.fusion_gate(c + t)
        fused = gate * c + (1 - gate) * t

        attn, _ = self.time_attention(fused, fused, fused)
        fused = self.norm_time(attn + fused)
        fused = self.drop_path(self.regime_gate(fused) * fused)

        context = fused.mean(dim=1)
        return self.fc(context)


class TCFNPlus3(nn.Module):
    """Variant that re-embeds transformer tokens and uses gated attention pooling."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        hidden_dim: int = 256,
        embedding_dim: int = 24,
        dropout: float = 0.2,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embed_features = embed_features
        self.encode_features = encode_features

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cat, embedding_dim) for num_cat in num_embed_features]
        )
        emb_dim_total = embedding_dim * len(embed_features)
        self.cont_features = _build_continuous_features(
            num_features, embed_features, encode_features
        )
        cont_dim = len(self.cont_features)

        self.time_encoder = LearnableTimeEncoder()
        enc_dim = 2 * len(encode_features)
        self.input_dim = emb_dim_total + cont_dim + enc_dim

        self.var_selection = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.Sigmoid()
        )

        self.cnn_branch = MultiScaleCNN(self.input_dim, hidden_dim, dropout)
        cnn_dim = 4 * hidden_dim
        self.reembed = nn.Linear(self.input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)

        self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
        self.tr_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.pool = GatedAttentionPooling(hidden_dim)
        self.drop_path = DropPath(0.1)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        parts: List[Tensor] = []
        if self.encode_features:
            parts.append(self.time_encoder(x[:, :, self.encode_features]))
        if self.embed_features:
            parts.append(
                torch.cat(
                    [
                        emb(x[:, :, feat].long())
                        for feat, emb in zip(self.embed_features, self.embeddings)
                    ],
                    dim=-1,
                )
            )
        if self.cont_features:
            parts.append(x[:, :, self.cont_features])

        z = torch.cat(parts, dim=-1)
        z = z * self.var_selection(z)

        cnn_out = self.cnn_branch(z)
        tr_out = self.transformer(self.reembed(z))

        c = self.cnn_proj(cnn_out)
        t = self.tr_proj(tr_out)
        fusion_gate = self.fusion_gate(c + t)
        fused = fusion_gate * c + (1 - fusion_gate) * t
        fused = self.drop_path(fused)

        context = self.pool(fused)
        return self.fc(context)


class TCFNPlus4(nn.Module):
    """Latest dual-branch model with cross-gated fusion and depthwise CNN."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_features: List[int],
        num_embed_features: List[int],
        encode_features: List[int],
        hidden_dim: int = 256,
        embedding_dim: int = 24,
        dropout: float = 0.2,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embed_features = embed_features
        self.encode_features = encode_features

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_cat, embedding_dim) for num_cat in num_embed_features]
        )
        emb_dim = embedding_dim * len(embed_features)
        self.cont_features = _build_continuous_features(
            num_features, embed_features, encode_features
        )
        cont_dim = len(self.cont_features)

        self.time_encoder = LearnableTimeEncoder()
        enc_dim = 2 * len(encode_features)
        self.input_dim = emb_dim + cont_dim + enc_dim

        self.var_gate = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim), nn.Sigmoid()
        )

        self.cnn = DualDepthwiseCNN(self.input_dim, hidden_dim)
        self.reembed = nn.Linear(self.input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)
        self.cross = CrossGate(hidden_dim)
        self.drop_path = DropPath(0.1)
        self.pool = AttentionPool(hidden_dim)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        parts: List[Tensor] = []
        if self.encode_features:
            parts.append(self.time_encoder(x[:, :, self.encode_features]))
        if self.embed_features:
            parts.append(
                torch.cat(
                    [
                        emb(x[:, :, feat].long())
                        for feat, emb in zip(self.embed_features, self.embeddings)
                    ],
                    dim=-1,
                )
            )
        if self.cont_features:
            parts.append(x[:, :, self.cont_features])

        z = torch.cat(parts, dim=-1)
        z = z * self.var_gate(z)

        cnn_latent = self.cnn(z)
        tr_latent = self.transformer(self.reembed(z))

        fused = self.drop_path(self.cross(cnn_latent, tr_latent))
        context = self.pool(fused)
        return self.fc(context)
