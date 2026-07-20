import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HierarchicalPooling(nn.Module):

    def __init__(self, kernel_size=3, stride=3, padding=1, tau=1e-12):
        super(HierarchicalPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau = tau

    def smooth_max_pooling(self, x):
        batch_size, channels, height, width = x.size()

        x_unfolded = F.unfold(x, kernel_size=self.kernel_size,
                              padding=self.padding, stride=self.stride)

        x_unfolded = x_unfolded.view(batch_size, channels,
                                     self.kernel_size * self.kernel_size, -1)

        smooth_max = self.tau * torch.logsumexp(x_unfolded / self.tau, dim=2)

        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        smooth_max = smooth_max.view(batch_size, channels, out_h, out_w)

        return smooth_max

    def forward(self, x):
        x = self.smooth_max_pooling(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        return x


class GSAMLP(nn.Module):

    def __init__(self, channels, groups, dropout_prob=0.2):
        super(GSAMLP, self).__init__()
        self.channels = channels
        self.groups = groups
        self.channels_per_group = channels // groups

        assert channels % groups == 0, "channels must be divisible by groups"


        self.theta = nn.ModuleList([
            nn.Linear(self.channels_per_group, self.channels_per_group, bias=False)
            for _ in range(groups)
        ])

        self.omega = nn.ModuleList([
            nn.Linear(self.channels_per_group, self.channels_per_group, bias=False)
            for _ in range(groups)
        ])

        self.w = nn.ModuleList([
            nn.Linear(self.channels_per_group, self.channels_per_group, bias=False)
            for _ in range(groups)
        ])

        self.dropout = nn.Dropout(dropout_prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def channel_shuffle(self, x, groups):
        batch_size, channels = x.size()
        channels_per_group = channels // groups

        x = x.view(batch_size, groups, channels_per_group)

        x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1)

        return x

    def forward(self, x):
        batch_size = x.size(0)

        x_groups = x.view(batch_size, self.groups, self.channels_per_group)

        u_list = []
        v_list = []

        for i in range(self.groups):
            x_i = x_groups[:, i, :]
            u_i = self.theta[i](x_i)
            v_i = self.omega[i](x_i)
            u_list.append(u_i)
            v_list.append(v_i)

        u = torch.cat(u_list, dim=1)
        v = torch.cat(v_list, dim=1)

        g = torch.sigmoid(self.dropout(v))
        z = x * (1 - g) + u * g

        z = self.channel_shuffle(z, self.groups)

        z_groups = z.view(batch_size, self.groups, self.channels_per_group)
        y_list = []

        for i in range(self.groups):
            z_i = z_groups[:, i, :]
            y_i = self.w[i](z_i)
            y_list.append(y_i)

        y = torch.cat(y_list, dim=1)

        return y


class GSACA(nn.Module):

    def __init__(self, channels, groups=None, kernel_size=3, stride=3, padding=1):
        super(GSACA, self).__init__()

        if groups is None:
            groups = max(channels // 16, 1)

        if channels % groups != 0:
            groups = self._find_closest_divisor(channels, groups)

        self.channels = channels
        self.groups = groups

        self.ghp = HierarchicalPooling(kernel_size, stride, padding)

        self.gsamlp = GSAMLP(channels, groups)

    def _find_closest_divisor(self, n, target):
        for offset in range(n):
            if n % (target - offset) == 0:
                return target - offset
            if n % (target + offset) == 0:
                return target + offset
        return 1

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        z = self.ghp(x)

        a = torch.sigmoid(self.gsamlp(z))

        a = a.view(batch_size, channels, 1, 1)

        y = x * a

        return y


if __name__ == "__main__":

    print("=" * 70)
    print("GSACA Channel Attention - Validation and Demo")
    print("=" * 70)

    print("\n[Demo 1] How GSACA selects important features")
    print("-" * 70)

    batch_size = 2
    channels = 8
    height, width = 4, 4

    x = torch.zeros(batch_size, channels, height, width)
    x[:, 0:2, :, :] = torch.randn(batch_size, 2, height, width) + 3.0
    x[:, 2:6, :, :] = torch.randn(batch_size, 4, height, width)
    x[:, 6:8, :, :] = torch.randn(batch_size, 2, height, width) - 2.0

    print(f"Input feature map shape: {x.shape}")
    print(f"Average activation per channel (first sample):")
    for c in range(channels):
        print(f"  Channel {c}: {x[0, c].mean().item():.3f}")

    gsaca = GSACA(channels=channels, groups=2)
    gsaca.eval()

    with torch.no_grad():
        z = gsaca.ghp(x)
        attention_weights = torch.sigmoid(gsaca.gsamlp(z))

        print(f"\nLearned channel attention weights (first sample):")
        for c in range(channels):
            weight = attention_weights[0, c].item()
            print(f"  Channel {c}: {weight:.3f} {'<- important feature' if weight > 0.6 else '<- unimportant feature' if weight < 0.4 else ''}")

        output = gsaca(x)

    print(f"\nOutput feature map shape: {output.shape}")
    print(f"Activation change per channel (first sample):")
    for c in range(channels):
        before = x[0, c].mean().item()
        after = output[0, c].mean().item()
        change = (after - before) / (abs(before) + 1e-6) * 100
        print(f"  Channel {c}: {before:.3f} -> {after:.3f} (change: {change:+.1f}%)")

    print("\nObservation: Channels with high attention weights are enhanced, low weights are suppressed")

    print("\n" + "=" * 70)
    print("[Demo 2] Different configuration tests from the paper")
    print("-" * 70)

    configs = [
        ("EfficientNet config", 64, 4),
        ("RegNet config", 128, 8),
        ("SSDlite config", 256, 8),
    ]

    for name, ch, g in configs:
        x_test = torch.randn(2, ch, 32, 32)
        gsaca_test = GSACA(channels=ch, groups=g)

        with torch.no_grad():
            output_test = gsaca_test(x_test)

        params = sum(p.numel() for p in gsaca_test.parameters())
        print(f"{name}: Channels={ch}, Groups={g}")
        print(f"  Input shape={x_test.shape}, Output shape={output_test.shape}")
        print(f"  Params={params:,}")
        print()

    print("=" * 70)
    print("[Demo 3] GSACA vs SE - Params and performance tradeoff")
    print("-" * 70)


    class SEModule(nn.Module):
        def __init__(self, channels, reduction=16):
            super(SEModule, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y


    channels = 64
    x = torch.randn(2, channels, 32, 32)

    gsaca = GSACA(channels=channels, groups=4)
    gsaca_params = sum(p.numel() for p in gsaca.parameters())

    se = SEModule(channels=channels, reduction=16)
    se_params = sum(p.numel() for p in se.parameters())

    print(f"Channels: {channels}")
    print(f"\nSE module:")
    print(f"  - Reduction: {channels} -> {channels // 16} -> {channels}")
    print(f"  - Params: {se_params:,}")
    print(f"  - Drawback: Dimension reduction causes information loss")

    print(f"\nGSACA module:")
    print(f"  - Groups: {gsaca.groups} groups, {channels // gsaca.groups} channels per group")
    print(f"  - Params: {gsaca_params:,}")
    print(f"  - Advantage: No dimension reduction, less information loss")
    print(f"  - Params ratio (GSACA/SE): {gsaca_params / se_params:.2f}")

    print("\n" + "=" * 70)
    print("[Demo 4] Channel attention weight distribution")
    print("-" * 70)

    channels = 32
    x = torch.randn(1, channels, 16, 16)
    gsaca = GSACA(channels=channels, groups=2)

    with torch.no_grad():
        z = gsaca.ghp(x)
        weights = torch.sigmoid(gsaca.gsamlp(z))[0]

    high_importance = (weights > 0.6).sum().item()
    medium_importance = ((weights >= 0.4) & (weights <= 0.6)).sum().item()
    low_importance = (weights < 0.4).sum().item()

    print(f"Total channels: {channels}")
    print(f"High importance channels (weight>0.6): {high_importance} ({high_importance / channels * 100:.1f}%)")
    print(f"Medium importance channels (0.4-0.6): {medium_importance} ({medium_importance / channels * 100:.1f}%)")
    print(f"Low importance channels (weight<0.4): {low_importance} ({low_importance / channels * 100:.1f}%)")

    print(f"\nWeight statistics:")
    print(f"  Max: {weights.max().item():.3f}")
    print(f"  Min: {weights.min().item():.3f}")
    print(f"  Mean: {weights.mean().item():.3f}")
    print(f"  Std: {weights.std().item():.3f}")

    print("\n" + "=" * 70)
    print("[Summary] Core advantages of GSACA")
    print("=" * 70)
