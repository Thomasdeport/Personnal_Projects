import torch
import torch.nn as nn
import numpy as np
from typing import List


class AdvancedRNN(nn.Module):
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
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(AdvancedRNN, self).__init__()

        self.device = device
        self.embed_features = embed_features
        self.num_embed_features = num_embed_features
        self.encode_features = encode_features
        self.attention = attention
        self.use_transformer = use_transformer

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embedding_dim) for num_classes in num_embed_features
        ])

        # Identify continuous features that are not embedded or encoded
        self.features_not_modified = [
            i for i in range(num_features) if i not in embed_features and i not in encode_features
        ]

        # Input dimension: embeddings + encoded features + continuous features
        self.input_dim = embedding_dim * len(embed_features) + len(encode_features) + len(self.features_not_modified)

        # Positional encoding (learned)
        self.positional_encoding = nn.Parameter(torch.randn(1, 500, self.input_dim))  # max seq length = 500

        # Transformer encoder
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=nhead,
                dim_feedforward=2 * self.input_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_transformer_layers
            )

        # RNN
        rnn_layer = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_layer(
            self.input_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        # Attention mechanism
        if attention:
            self.attention_weights = nn.Linear(2 * hidden_dim, 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Encoding function (sinusoidal fallback)
        self.encoder = lambda u: torch.sin(torch.pi * u / 100)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device

        inputs = []

        # Encoded features
        if self.encode_features:
            encoded = self.encoder(x[:, :, self.encode_features])
            inputs.append(encoded)

        # Embedded categorical features
        if self.embed_features:
            embedded = []
            for i, idx in enumerate(self.embed_features):
                embedded.append(self.embeddings[i](x[:, :, idx].long()))
            inputs.append(torch.cat(embedded, dim=-1))

        # Continuous features
        if self.features_not_modified:
            inputs.append(x[:, :, self.features_not_modified])

        # Concatenate all inputs
        input_tensor = torch.cat(inputs, dim=-1)

        # Add positional encoding
        seq_len = input_tensor.size(1)
        input_tensor = input_tensor + self.positional_encoding[:, :seq_len, :].to(device)

        # Transformer
        if self.use_transformer:
            input_tensor = self.transformer_encoder(input_tensor)

        # RNN
        rnn_out, _ = self.rnn(input_tensor)

        # Attention
        if self.attention:
            scores = torch.softmax(self.attention_weights(rnn_out), dim=1)
            context_vector = torch.sum(scores * rnn_out, dim=1)
        else:
            context_vector = torch.mean(rnn_out, dim=1)

        return self.fc(context_vector)

    def __str__(self):
        return (
            f"AdvancedRNN(\n"
            f"  Input Dim: {self.input_dim}\n"
            f"  Hidden Dim: {self.hidden_dim}\n"
            f"  Attention: {self.attention}\n"
            f"  Transformer: {self.use_transformer}\n"
            f"  Device: {self.device}\n"
            f")"
        )



class gru_lstm(nn.Module):
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
        rnn_type: str = "GRU",  # Choice between GRU and LSTM
        attention: bool = True,  # Enable attention mechanism
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(gru_lstm, self).__init__()

        # Store model parameters
        self.num_features = num_features
        self.num_class = num_class
        self.embed_features = embed_features
        self.num_embed_features = num_embed_features
        self.encode_features = encode_features
        self.embedding_dim = embedding_dim
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.attention = attention
        self.device = device

        # Identify features that are neither embedded nor encoded
        self.features_not_modified = [
            i for i in range(self.num_features) if i not in self.embed_features and i not in self.encode_features
        ]

        # Define embedding layer for categorical features
        self.embeddings = nn.Embedding(sum(self.num_embed_features), self.embedding_dim)
        self.embeddings_offset = torch.tensor(
            np.concatenate(([0], np.cumsum(self.num_embed_features)[:-1])),
            dtype=torch.int,
            device=self.device,
        )

        # Compute input dimension after embedding and encoding
        self.input_dim = sum(
            [self.embedding_dim if i in self.embed_features else 1 for i in range(self.num_features)]
        )

        # Define RNN layer (either GRU or LSTM)
        rnn_layer = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_layer(
            self.input_dim,
            self.d_hidden,
            self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Define attention mechanism if enabled
        if self.attention:
            self.attention_weights = nn.Linear(2 * self.d_hidden, 1)

        # Fully connected layers for classification output
        self.fc = nn.Sequential(
            nn.LayerNorm(2 * self.d_hidden),  # Normalize RNN outputs
            nn.Linear(2 * self.d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, self.num_class),
        )

        # Feature encoder for continuous variables using sinusoidal transformation
        self.encoder = lambda u: torch.sin(torch.pi * u / 100)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Initialize input tensor
        input = torch.zeros(batch_size, seq_len, self.input_dim, device=x.device)
        count = 0

        # Encode selected features
        input[:, :, count : count + len(self.encode_features)] = self.encoder(x[:, :, self.encode_features])
        count += len(self.encode_features)

        # Embed categorical features
        input[:, :, count : count + self.embedding_dim * len(self.embed_features)] = (
            self.embeddings(x[:, :, self.embed_features].type(torch.int) + self.embeddings_offset)
        ).reshape(batch_size, seq_len, self.embedding_dim * len(self.embed_features))
        count += self.embedding_dim * len(self.embed_features)

        # Copy unmodified continuous features
        input[:, :, count:] = x[:, :, self.features_not_modified]

        # Pass input through the RNN
        rnn_out, _ = self.rnn(input)

        # Apply attention mechanism if enabled
        if self.attention:
            attention_scores = torch.softmax(self.attention_weights(rnn_out), dim=1)
            context_vector = torch.sum(attention_scores * rnn_out, dim=1)
        else:
            # Use mean pooling as fallback
            context_vector = torch.mean(rnn_out, dim=1)

        # Pass through fully connected layers
        output = self.fc(context_vector)
        return output

    def __str__(self):
        return f"AdvancedModel(num_features={self.num_features}, num_class={self.num_class}, embed_features={self.embed_features}, num_embed_features={self.num_embed_features}, encode_features={self.encode_features}, embedding_dim={self.embedding_dim}, d_hidden={self.d_hidden}, num_layers={self.num_layers}, attention={self.attention})"
