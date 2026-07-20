import torch
import torch.nn as nn


class TemporalMemoryModule(nn.Module):
    """Compute the 3-day and 7-day hydrological memory in Equation (4)."""

    def __init__(self, short_memory: int = 3, long_memory: int = 7):
        super().__init__()
        if short_memory <= 0 or long_memory <= 0:
            raise ValueError("Memory windows must be positive")
        self.short_memory = int(short_memory)
        self.long_memory = int(long_memory)

    @staticmethod
    def _rolling_mean(series: torch.Tensor, window: int) -> torch.Tensor:
        batch_size, length = series.shape
        prefix = torch.cat(
            [series.new_zeros(batch_size, 1), torch.cumsum(series, dim=1)],
            dim=1,
        )
        end = torch.arange(1, length + 1, device=series.device)
        start = torch.clamp(end - window, min=0)
        total = prefix[:, end] - prefix[:, start]
        count = (end - start).to(series.dtype).unsqueeze(0)
        return total / count

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.size(-1) < 3:
            raise ValueError("Gate input must have shape [batch, time, 3]")

        pcp = x[..., 0]
        sw = x[..., 1]
        pet = x[..., 2]

        return {
            "pcp_3d": self._rolling_mean(pcp, self.short_memory),
            "pcp_7d": self._rolling_mean(pcp, self.long_memory),
            "sw_7d": self._rolling_mean(sw, self.long_memory),
            "pet_7d": self._rolling_mean(pet, self.long_memory),
        }
