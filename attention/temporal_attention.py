
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):

    def __init__(self, tcn_filters):
        super(TemporalAttention, self).__init__()

        self.tcn_filters = tcn_filters

        self.attention_net = nn.Sequential(
            nn.Linear(tcn_filters, tcn_filters // 2),
            nn.Tanh(),
            nn.Linear(tcn_filters // 2, 1)
        )

    def forward(self, x):
        attn_scores = self.attention_net(x)

        beta = F.softmax(attn_scores, dim=1)

        weighted_out = x * beta

        return weighted_out, beta
