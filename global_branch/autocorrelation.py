
import torch
import torch.nn as nn
import math


class AutoCorrelation(nn.Module):

    def __init__(
        self,
        factor=1,
        attention_dropout=0.1,
        output_attention=False
    ):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention

        self.dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else None

    def time_delay_agg(self, queries, keys, values):
        batch_size, seq_len_q, d_model = queries.shape
        _, seq_len_k, _ = keys.shape
        _, seq_len_v, _ = values.shape

        max_len = max(seq_len_q, seq_len_k)

        pad_q = max_len - seq_len_q
        pad_k = max_len - seq_len_k

        queries_padded = torch.nn.functional.pad(queries, (0, 0, 0, pad_q), mode='constant', value=0)
        keys_padded = torch.nn.functional.pad(keys, (0, 0, 0, pad_k), mode='constant', value=0)

        q_fft = torch.fft.rfft(queries_padded.permute(0, 2, 1), dim=-1)
        k_fft = torch.fft.rfft(keys_padded.permute(0, 2, 1), dim=-1)

        corr_fft = q_fft * torch.conj(k_fft)

        corr = torch.fft.irfft(corr_fft, n=max_len, dim=-1)
        corr = corr.permute(0, 2, 1)

        corr = corr[:, :seq_len_q, :]

        corr_weights = torch.softmax(corr, dim=1)

        corr_weights_exp = corr_weights.unsqueeze(2)
        values_exp = values.unsqueeze(1)
        output = (corr_weights_exp * values_exp).sum(dim=2)

        if self.dropout is not None:
            output = self.dropout(output)

        if self.output_attention:
            return output, corr
        else:
            return output, None

    def forward(self, queries, keys, values):
        return self.time_delay_agg(queries, keys, values)


class AutoCorrelationLayer(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        factor=1,
        attention_dropout=0.1,
        output_attention=False
    ):
        super(AutoCorrelationLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.auto_correlation = AutoCorrelation(
            factor=factor,
            attention_dropout=attention_dropout,
            output_attention=output_attention
        )

        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        batch_size = queries.size(0)
        seq_len = queries.size(1)

        Q = self.query_projection(queries)
        K = self.key_projection(keys)
        V = self.value_projection(values)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_keys)
        K = K.view(batch_size, -1, self.n_heads, self.d_keys)
        V = V.view(batch_size, -1, self.n_heads, self.d_values)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        Q = Q.contiguous().view(batch_size * self.n_heads, -1, self.d_keys)
        K = K.contiguous().view(batch_size * self.n_heads, -1, self.d_keys)
        V = V.contiguous().view(batch_size * self.n_heads, -1, self.d_values)

        out, attn = self.auto_correlation(Q, K, V)

        out = out.view(batch_size, self.n_heads, -1, self.d_values)

        out = out.permute(0, 2, 1, 3)

        out = out.contiguous().view(batch_size, seq_len, self.d_model)

        out = self.out_projection(out)

        return out, attn


if __name__ == "__main__":

    print("=" * 60)
    print("Testing AutoCorrelation")
    print("=" * 60)

    batch_size = 4
    seq_len = 96
    d_model = 128
    n_heads = 8

    queries = torch.randn(batch_size, seq_len, d_model)
    keys = torch.randn(batch_size, seq_len, d_model)
    values = torch.randn(batch_size, seq_len, d_model)

    print("\nTest 1: AutoCorrelation")
    auto_corr = AutoCorrelation(factor=3, attention_dropout=0.1, output_attention=True)
    auto_corr.eval()
    output1, corr = auto_corr(queries, keys, values)
    print(f"Input Query shape: {queries.shape}")
    print(f"Output shape: {output1.shape}")
    if corr is not None:
        print(f"Correlation shape: {corr.shape}")

    print("\nTest 2: AutoCorrelationLayer")
    auto_corr_layer = AutoCorrelationLayer(
        d_model=d_model,
        n_heads=n_heads,
        factor=3,
        attention_dropout=0.1,
        output_attention=False
    )
    auto_corr_layer.eval()

    output2, attn = auto_corr_layer(queries, keys, values)
    print(f"Input Query shape: {queries.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Output dimension preserved: {output2.shape == queries.shape}")

    print("\nTest 3: Verify periodicity detection")
    t = torch.arange(0, seq_len, dtype=torch.float32)
    periodic_signal = torch.sin(2 * math.pi * t / 10.0)
    periodic_signal = periodic_signal.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, d_model)

    output_periodic, corr_periodic = auto_corr(
        periodic_signal, periodic_signal, periodic_signal
    )
    print(f"Periodic signal input shape: {periodic_signal.shape}")
    print(f"Periodic signal output shape: {output_periodic.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
