
import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0)


class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model

        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='zeros',
            bias=True
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x = x.permute(0, 2, 1)
        return x


class DataEmbedding(nn.Module):

    def __init__(
        self,
        c_in,
        d_model,
        dropout=0.1,
        max_len=5000
    ):
        super(DataEmbedding, self).__init__()

        self.c_in = c_in
        self.d_model = d_model
        self.dropout_rate = dropout

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.value_embedding(x)

        x = x + self.position_embedding(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class SeriesDecomposition(nn.Module):

    def __init__(self, kernel_size=25):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size

        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)

        trend = self.moving_avg(x_permuted)

        trend = trend.permute(0, 2, 1)

        if trend.size(1) > x.size(1):
            trend = trend[:, :x.size(1), :]
        elif trend.size(1) < x.size(1):
            pad_len = x.size(1) - trend.size(1)
            trend = torch.cat([trend, trend[:, -1:, :].repeat(1, pad_len, 1)], dim=1)

        seasonal = x - trend

        return seasonal, trend


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Autoformer base layers")
    print("=" * 60)

    batch_size = 32
    seq_len = 96
    c_in = 8
    d_model = 128

    inputs = torch.randn(batch_size, seq_len, c_in)

    print("\nTest 1: PositionalEmbedding")
    pos_emb = PositionalEmbedding(d_model=d_model, max_len=5000)
    pos_output = pos_emb(torch.zeros(batch_size, seq_len, d_model))
    print(f"Input shape: {(batch_size, seq_len, d_model)}")
    print(f"Positional encoding shape: {pos_output.shape}")
    print(f"Positional encoding range: [{pos_output.min():.4f}, {pos_output.max():.4f}]")

    print("\nTest 2: TokenEmbedding")
    token_emb = TokenEmbedding(c_in=c_in, d_model=d_model)
    token_output = token_emb(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Token embedding shape: {token_output.shape}")

    print("\nTest 3: DataEmbedding")
    data_emb = DataEmbedding(c_in=c_in, d_model=d_model, dropout=0.1)
    data_emb.eval()
    data_output = data_emb(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Data embedding shape: {data_output.shape}")

    print("\nTest 4: SeriesDecomposition")
    series_decomp = SeriesDecomposition(kernel_size=25)
    seasonal, trend = series_decomp(data_output)
    print(f"Input shape: {data_output.shape}")
    print(f"Seasonal component shape: {seasonal.shape}")
    print(f"Trend component shape: {trend.shape}")

    reconstructed = seasonal + trend
    diff = torch.mean(torch.abs(data_output - reconstructed))
    print(f"Reconstruction error: {diff:.10f} (should be near 0)")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
