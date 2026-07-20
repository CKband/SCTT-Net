import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    """Fuse global and local features using Equation (7)."""

    def forward(
        self,
        gate: torch.Tensor,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
    ) -> torch.Tensor:
        if global_features.shape != local_features.shape:
            raise ValueError("Global and local feature shapes must match")
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1)
        if gate.size(1) == 1 and global_features.size(1) > 1:
            gate = gate.expand(-1, global_features.size(1), -1)
        if gate.shape[:2] != global_features.shape[:2]:
            raise ValueError("Gate and branch time dimensions must match")

        return gate * global_features + (1.0 - gate) * local_features
