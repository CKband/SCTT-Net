import torch
import torch.nn as nn

from ..gate.adaptive_fusion import AdaptiveFusion
from ..gate.gate_controller import GateController
from ..global_branch.transformer import TransformerBranch
from ..local_branch.tcn import TCN


class SCTTNet(nn.Module):
    """Manuscript-aligned implementation of SCTT-Net."""

    def __init__(
        self,
        num_meteo_features: int,
        num_gate_features: int = 3,
        pred_len: int = 1,
        seq_len: int = 30,
        label_len: int = 15,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        tcn_channels: list[int] | None = None,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        gate_hidden_dims: list[int] | None = None,
        gate_dropout: float = 0.1,
        gate_short_memory: int = 3,
        gate_long_memory: int = 7,
        gate_current_ratio: float = 0.7,
        gate_memory_ratio: float = 0.3,
        gate_theta_init: float = 0.0,
        gate_w1: float = 1.5,
        gate_w2: float = 1.0,
        gate_w3: float = 1.0,
        gate_w4: float = 1.0,
        gate_w5: float = 0.8,
        gate_w6: float = 0.5,
        gate_w7: float = 0.6,
        model_variant: str = "SCTT-Net",
    ):
        super().__init__()

        tcn_channels = [d_model, d_model, d_model] if tcn_channels is None else list(tcn_channels)
        gate_hidden_dims = [32, 16] if gate_hidden_dims is None else list(gate_hidden_dims)

        self.num_meteo_features = int(num_meteo_features)
        self.num_gate_features = int(num_gate_features)
        self.pred_len = int(pred_len)
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.d_model = int(d_model)
        self.model_variant = model_variant

        self.global_branch = TransformerBranch(
            num_features=num_meteo_features,
            d_model=d_model,
            n_heads=n_heads,
            encoder_layers=e_layers,
            decoder_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        self.local_input_projection = nn.Linear(num_meteo_features, d_model)
        self.local_branch = TCN(
            num_inputs=d_model,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout_rate=tcn_dropout,
        )
        self.local_output_projection = nn.Linear(tcn_channels[-1], d_model)
        self.local_horizon_projection = nn.Linear(d_model, pred_len * d_model)

        self.gate_controller = GateController(
            num_physical_vars=num_gate_features,
            hidden_dims=gate_hidden_dims,
            dropout_rate=gate_dropout,
            short_memory=gate_short_memory,
            long_memory=gate_long_memory,
            current_ratio=gate_current_ratio,
            memory_ratio=gate_memory_ratio,
            theta_init=gate_theta_init,
            w1=gate_w1,
            w2=gate_w2,
            w3=gate_w3,
            w4=gate_w4,
            w5=gate_w5,
            w6=gate_w6,
            w7=gate_w7,
        )
        self.adaptive_fusion = AdaptiveFusion()
        self.final_projection = nn.Linear(d_model, 1)

    def _prepare_inputs(self, meteo_input):
        if isinstance(meteo_input, dict):
            return meteo_input["enc_x"], meteo_input["dec_x"]

        enc_x = meteo_input[:, : self.seq_len, :]
        historical = enc_x[:, -self.label_len :, :]
        future = enc_x.new_zeros(
            enc_x.size(0), self.pred_len, self.num_meteo_features
        )
        return enc_x, torch.cat([historical, future], dim=1)

    def forward(
        self,
        meteo_input,
        gate_input: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        enc_x, dec_x = self._prepare_inputs(meteo_input)

        global_sequence = self.global_branch(enc_x, dec_x)
        global_features = global_sequence[:, -self.pred_len :, :]

        local_sequence = self.local_branch(self.local_input_projection(enc_x))
        local_state = self.local_output_projection(local_sequence[:, -1, :])
        local_features = self.local_horizon_projection(local_state).view(
            enc_x.size(0), self.pred_len, self.d_model
        )

        gate_sequence, gate_components = self.gate_controller(
            gate_input, return_components=True
        )
        gate = gate_sequence[:, -1:, :].expand(-1, self.pred_len, -1)

        variant = self.model_variant.lower()
        if variant == "transformer":
            fused_features = global_features
        elif variant == "tcn":
            fused_features = local_features
        elif variant == "mg":
            basic_gate = gate_components["g_mlp"][:, -1:, :].expand(
                -1, self.pred_len, -1
            )
            fused_features = self.adaptive_fusion(
                basic_gate, global_features, local_features
            )
            gate = basic_gate
        else:
            fused_features = self.adaptive_fusion(
                gate, global_features, local_features
            )

        output = self.final_projection(fused_features)
        if not return_diagnostics:
            return output

        return output, {
            "gate": gate,
            "global_features": global_features,
            "local_features": local_features,
            **gate_components,
        }


def build_sctt_net_from_config(config):
    return SCTTNet(
        num_meteo_features=config.num_meteo_features,
        num_gate_features=config.num_gate_features,
        pred_len=config.PRED_LEN,
        seq_len=config.SEQ_LEN,
        label_len=config.LABEL_LEN,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        e_layers=config.E_LAYERS,
        d_layers=config.D_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        activation=config.ACTIVATION,
        tcn_channels=config.TCN_CHANNELS,
        tcn_kernel_size=config.TCN_KERNEL_SIZE,
        tcn_dropout=config.TCN_DROPOUT,
        gate_hidden_dims=config.GATE_HIDDEN_DIMS,
        gate_dropout=config.GATE_DROPOUT,
        gate_short_memory=config.GATE_SHORT_MEMORY,
        gate_long_memory=config.GATE_LONG_MEMORY,
        gate_current_ratio=config.GATE_CURRENT_RATIO,
        gate_memory_ratio=config.GATE_MEMORY_RATIO,
        gate_theta_init=config.GATE_THETA_INIT,
        gate_w1=config.GATE_W1,
        gate_w2=config.GATE_W2,
        gate_w3=config.GATE_W3,
        gate_w4=config.GATE_W4,
        gate_w5=config.GATE_W5,
        gate_w6=config.GATE_W6,
        gate_w7=config.GATE_W7,
        model_variant=config.MODEL_VARIANT,
    )
