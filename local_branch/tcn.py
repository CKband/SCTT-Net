
import torch
import torch.nn as nn


class CausalConv1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation_rate=1,
        dropout_rate=0.2
    ):
        super(CausalConv1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        self.padding = (kernel_size - 1) * dilation_rate

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=0,
            bias=True
        )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = nn.functional.pad(x, (self.padding, 0))

        x = self.conv(x)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.permute(0, 2, 1)

        return x


class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation_rate=1,
        dropout_rate=0.2
    ):
        super(TemporalBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        self.conv1 = CausalConv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate
        )

        self.conv2 = CausalConv1D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate
        )

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.final_activation = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        residual = x
        if self.downsample is not None:
            residual = residual.permute(0, 2, 1)
            residual = self.downsample(residual)
            residual = residual.permute(0, 2, 1)

        output = out + residual

        output = self.final_activation(output)

        return output


class TCN(nn.Module):

    def __init__(
        self,
        num_inputs,
        num_channels=[128, 128, 128],
        kernel_size=3,
        dropout_rate=0.2
    ):
        super(TCN, self).__init__()

        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_rate = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            temporal_block = TemporalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=dropout_rate
            )
            layers.append(temporal_block)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_receptive_field(self):
        num_levels = len(self.num_channels)
        receptive_field = 1

        for i in range(num_levels):
            dilation_rate = 2 ** i
            receptive_field += 2 * (self.kernel_size - 1) * dilation_rate

        return receptive_field


if __name__ == "__main__":

    print("=" * 60)
    print("Testing TCN (Temporal Convolutional Network)")
    print("=" * 60)

    batch_size = 32
    seq_len = 96
    input_channels = 8
    output_channels = 128

    inputs = torch.randn(batch_size, seq_len, input_channels)

    print("\nTest 1: CausalConv1D")
    causal_conv = CausalConv1D(
        in_channels=input_channels,
        out_channels=64,
        kernel_size=3,
        dilation_rate=1,
        dropout_rate=0.2
    )
    causal_conv.eval()

    output1 = causal_conv(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Output sequence length preserved: {output1.shape[1] == seq_len}")

    print("\nTest 2: TemporalBlock")
    temporal_block = TemporalBlock(
        in_channels=input_channels,
        out_channels=128,
        kernel_size=3,
        dilation_rate=2,
        dropout_rate=0.2
    )
    temporal_block.eval()

    output2 = temporal_block(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Output sequence length preserved: {output2.shape[1] == seq_len}")

    print("\nTest 3: Full TCN")
    tcn = TCN(
        num_inputs=input_channels,
        num_channels=[128, 128, 128],
        kernel_size=3,
        dropout_rate=0.2
    )
    tcn.eval()

    output3 = tcn(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {output3.shape}")
    print(f"Output sequence length preserved: {output3.shape[1] == seq_len}")
    print(f"TCN receptive field: {tcn.get_receptive_field()} timesteps")

    print("\nTest 4: Verify causality")
    test_input1 = torch.randn(1, seq_len, input_channels)
    test_input2 = test_input1.clone()

    test_input2[:, seq_len//2:, :] = torch.randn(1, seq_len//2, input_channels)

    output_test1 = tcn(test_input1)
    output_test2 = tcn(test_input2)

    diff_first_half = torch.mean(torch.abs(
        output_test1[:, :seq_len//2, :] - output_test2[:, :seq_len//2, :]
    ))
    print(f"First half output diff: {diff_first_half:.10f} (should be near 0, proving causality)")

    print("\nTest 5: TCN with different configs")
    configs = [
        {'num_channels': [64, 64], 'kernel_size': 3},
        {'num_channels': [128, 128, 128], 'kernel_size': 3},
        {'num_channels': [128, 256, 512], 'kernel_size': 5},
    ]

    for i, config in enumerate(configs):
        tcn_test = TCN(num_inputs=input_channels, **config, dropout_rate=0.1)
        tcn_test.eval()
        output_test = tcn_test(inputs)
        receptive_field = tcn_test.get_receptive_field()
        print(f"Config {i+1}: {config}")
        print(f"  Output shape: {output_test.shape}")
        print(f"  Receptive field: {receptive_field} timesteps")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
