import torch
import torch.nn as nn

from ..embedding.layers import DataEmbedding


class TransformerBranch(nn.Module):
    """Encoder-decoder Transformer used as the global SCTT-Net branch."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.encoder_embedding = DataEmbedding(num_features, d_model, dropout)
        self.decoder_embedding = DataEmbedding(num_features, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(self.encoder_embedding(enc_x))
        decoder_input = self.decoder_embedding(dec_x)
        mask = self._causal_mask(decoder_input.size(1), decoder_input.device)
        return self.decoder(decoder_input, memory, tgt_mask=mask)
