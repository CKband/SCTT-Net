import torch
import torch.nn as nn

from .temporal_memory import TemporalMemoryModule


class GateController(nn.Module):
    """Physical Gating Mechanism defined by Equations (1) to (6)."""

    def __init__(
        self,
        num_physical_vars: int = 3,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
        short_memory: int = 3,
        long_memory: int = 7,
        current_ratio: float = 0.7,
        memory_ratio: float = 0.3,
        theta_init: float = 0.0,
        w1: float = 1.5,
        w2: float = 1.0,
        w3: float = 1.0,
        w4: float = 1.0,
        w5: float = 0.8,
        w6: float = 0.5,
        w7: float = 0.6,
    ):
        super().__init__()

        if num_physical_vars != 3:
            raise ValueError("The manuscript gate requires PCP, SW, and PET")
        if abs(current_ratio + memory_ratio - 1.0) > 1e-6:
            raise ValueError("Current and memory ratios must sum to one")

        hidden_dims = [32, 16] if hidden_dims is None else list(hidden_dims)
        layers: list[nn.Module] = []
        input_dim = num_physical_vars
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        layers.extend([nn.Linear(input_dim, 1), nn.Sigmoid()])
        self.basic_gate = nn.Sequential(*layers)

        self.temporal_memory = TemporalMemoryModule(short_memory, long_memory)
        self.current_ratio = float(current_ratio)
        self.memory_ratio = float(memory_ratio)
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))

        self.register_buffer(
            "physical_weights",
            torch.tensor([w1, w2, w3, w4, w5, w6, w7], dtype=torch.float32),
        )

    def get_physics_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        squeeze_output = x.dim() == 2
        if squeeze_output:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError("Gate input must have shape [batch, time, 3]")

        pcp = x[..., 0]
        sw = x[..., 1]
        pet = x[..., 2]
        memory = self.temporal_memory(x)
        w = self.physical_weights.to(dtype=x.dtype)

        score_current = w[0] * pcp + w[1] * sw + w[2] * pet
        score_memory = (
            w[3] * memory["pcp_3d"]
            + w[4] * memory["pcp_7d"]
            + w[5] * memory["sw_7d"]
            + w[6] * memory["pet_7d"]
        )
        physics_raw = (
            self.current_ratio * score_current
            + self.memory_ratio * score_memory
        )
        g_phy = torch.sigmoid(physics_raw).unsqueeze(-1)

        components = {
            "score_current": score_current,
            "score_memory": score_memory,
            "physics_raw": physics_raw,
            "g_phy": g_phy,
            **memory,
        }
        if squeeze_output:
            components = {
                key: value.squeeze(1) for key, value in components.items()
            }
        return components

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ):
        squeeze_output = x.dim() == 2
        if squeeze_output:
            x = x.unsqueeze(1)

        g_mlp = self.basic_gate(x)
        components = self.get_physics_components(x)
        g_phy = components["g_phy"]
        lambda_weight = torch.sigmoid(self.theta)
        gate = (1.0 - lambda_weight) * g_mlp + lambda_weight * g_phy

        if squeeze_output:
            gate = gate.squeeze(1)
            g_mlp = g_mlp.squeeze(1)
            components = {
                key: value.squeeze(1)
                if isinstance(value, torch.Tensor)
                and value.dim() > 1
                and value.size(1) == 1
                else value
                for key, value in components.items()
            }

        if not return_components:
            return gate

        return gate, {
            **components,
            "g_mlp": g_mlp,
            "lambda": lambda_weight,
        }
