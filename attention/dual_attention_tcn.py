
import torch.nn as nn

from .feature_attention import FeatureAttention
from .temporal_attention import TemporalAttention


class DualAttentionTCN(nn.Module):

    def __init__(self, num_input_features, d_model, tcn_module, tcn_output_filters, output_dim):
        super(DualAttentionTCN, self).__init__()

        self.num_input_features = num_input_features
        self.d_model = d_model
        self.tcn_output_filters = tcn_output_filters
        self.output_dim = output_dim

        self.feature_attention = FeatureAttention(num_input_features)

        self.input_projection = nn.Linear(num_input_features, d_model)

        self.tcn = tcn_module

        self.temporal_attention = TemporalAttention(tcn_output_filters)

        self.output_projection = nn.Linear(tcn_output_filters, output_dim)

    def forward(self, x):
        weighted_input, alpha = self.feature_attention(x)

        projected_input = self.input_projection(weighted_input)

        tcn_output = self.tcn(projected_input)

        weighted_tcn, beta = self.temporal_attention(tcn_output)

        local_features = self.output_projection(weighted_tcn)

        return local_features, alpha, beta
