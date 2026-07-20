
import torch
import torch.nn as nn

from .GSACA import GSACA


class FeatureAttention(nn.Module):

    def __init__(self, num_features, groups=None):
        super(FeatureAttention, self).__init__()

        self.num_features = num_features

        self.gsaca = GSACA(channels=num_features, groups=groups)

    def forward(self, x):
        B, S, F = x.shape

        x_4d = x.permute(0, 2, 1).unsqueeze(-1)

        z = self.gsaca.ghp(x_4d)
        alpha_channel = torch.sigmoid(self.gsaca.gsamlp(z))

        alpha = alpha_channel.unsqueeze(1).expand(-1, S, -1)

        weighted_x = x * alpha

        return weighted_x, alpha
