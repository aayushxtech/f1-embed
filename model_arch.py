import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Calculate sinusoidal values for each dimension
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to embeddings
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class F1Embedder(nn.Module):
    def __init__(self, telemetry_dim, context_dim, d_model=512, nhead=4, num_layers=2):
        super().__init__()
        self.embedding_layer = nn.Linear(telemetry_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        self.regressor = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_context):
        x_seq = self.embedding_layer(x_seq)
        x_seq = self.pos_encoder(x_seq)
        z_seq = self.transformer(x_seq)
        z_seq = z_seq.mean(dim=1)

        z_context = self.context_mlp(x_context)

        combined = torch.cat([z_seq, z_context], dim=1)
        output = self.regressor(combined)
        return output.squeeze()
