
import torch
import torch.nn as nn
from .autocorrelation import AutoCorrelationLayer
from ..embedding.layers import SeriesDecomposition


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff=2048,
        moving_avg=25,
        dropout=0.1,
        activation='gelu',
        factor=1
    ):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.attention = AutoCorrelationLayer(
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            attention_dropout=dropout,
            output_attention=False
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.decomp1 = SeriesDecomposition(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomposition(kernel_size=moving_avg)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        new_x, _ = self.attention(x, x, x)
        x = x + self.dropout1(new_x)
        x = self.norm1(x)
        x, _ = self.decomp1(x)

        new_x = self.ffn(x)
        x = x + self.dropout2(new_x)
        x = self.norm2(x)
        x, _ = self.decomp2(x)

        return x


class Encoder(nn.Module):

    def __init__(self, encoder_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm_layer = norm_layer

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff=2048,
        moving_avg=25,
        dropout=0.1,
        activation='gelu',
        factor=1
    ):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.self_attention = AutoCorrelationLayer(
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            attention_dropout=dropout,
            output_attention=False
        )

        self.cross_attention = AutoCorrelationLayer(
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            attention_dropout=dropout,
            output_attention=False
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.decomp1 = SeriesDecomposition(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomposition(kernel_size=moving_avg)
        self.decomp3 = SeriesDecomposition(kernel_size=moving_avg)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, cross):
        new_x, _ = self.self_attention(x, x, x)
        x = x + self.dropout1(new_x)
        x = self.norm1(x)
        x, trend1 = self.decomp1(x)

        new_x, _ = self.cross_attention(x, cross, cross)
        x = x + self.dropout2(new_x)
        x = self.norm2(x)
        x, trend2 = self.decomp2(x)

        new_x = self.ffn(x)
        x = x + self.dropout3(new_x)
        x = self.norm3(x)
        x, trend3 = self.decomp3(x)

        residual_trend = trend1 + trend2 + trend3

        return x, residual_trend


class Decoder(nn.Module):

    def __init__(
        self,
        decoder_layers,
        norm_layer=None,
        projection=None
    ):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.norm_layer = norm_layer
        self.projection = projection

    def forward(self, x, cross):
        trend = 0.0

        for decoder_layer in self.decoder_layers:
            x, residual_trend = decoder_layer(x, cross)
            trend = trend + residual_trend

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        if self.projection is not None:
            x = self.projection(x)
            trend = self.projection(trend)

        return x, trend
