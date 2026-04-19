import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank_ratio=0.5, bias=True):
        super().__init__()
        rank = max(1, int(min(in_features, out_features) * rank_ratio))
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.up(self.down(x))


class LayerNormChannelsFirst(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class ConvEncoder(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, rank_ratio=0.5):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.pwconv1 = LoRaLin(dim, hidden, rank_ratio)
        self.act = nn.GELU()
        self.pwconv2 = LoRaLin(hidden, dim, rank_ratio)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class SDTA(nn.Module):
    def __init__(self, dim, num_heads=4, num_scales=3, mlp_ratio=4.0, rank_ratio=0.5):
        super().__init__()
        self.num_scales = num_scales
        split = dim // num_scales

        self.dw_convs = nn.ModuleList([
            nn.Conv2d(split, split, 3, 1, 1, groups=split) for _ in range(num_scales - 1)
        ])

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            LoRaLin(dim, hidden, rank_ratio),
            nn.GELU(),
            LoRaLin(hidden, dim, rank_ratio),
        )

    def forward(self, x):
        splits = list(x.chunk(self.num_scales, dim=1))
        for i in range(1, self.num_scales):
            splits[i] = self.dw_convs[i - 1](splits[i] + splits[i - 1])
        x = torch.cat(splits, dim=1)

        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        y = self.norm1(x_flat)
        qkv = self.qkv(y).reshape(B, H * W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 4, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        y = attn @ v
        y = y.permute(0, 3, 1, 2).reshape(B, H * W, C)
        y = self.proj(y)

        x_flat = x_flat + y
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return x_flat.transpose(1, 2).reshape(B, C, H, W)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNormChannelsFirst(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 2, 2, 0)

    def forward(self, x):
        return self.conv(self.norm(x))


class EdgeFace(nn.Module):
    def __init__(self, embedding_dim=512, dims=(48, 96, 160, 304), depths=(2, 2, 6, 2)):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 4, 4, 0),
            LayerNormChannelsFirst(dims[0]),
        )

        self.stage1 = nn.Sequential(*[ConvEncoder(dims[0]) for _ in range(depths[0])])

        self.down1 = DownsampleLayer(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[ConvEncoder(dims[1]) for _ in range(depths[1])])

        self.down2 = DownsampleLayer(dims[1], dims[2])
        stage3_blocks = [ConvEncoder(dims[2]) for _ in range(depths[2] - 1)]
        stage3_blocks.append(SDTA(dims[2], num_heads=4, num_scales=4))
        self.stage3 = nn.Sequential(*stage3_blocks)

        self.down3 = DownsampleLayer(dims[2], dims[3])
        stage4_blocks = [ConvEncoder(dims[3]) for _ in range(depths[3] - 1)]
        stage4_blocks.append(SDTA(dims[3], num_heads=4, num_scales=4))
        self.stage4 = nn.Sequential(*stage4_blocks)

        self.norm = LayerNormChannelsFirst(dims[3])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            LoRaLin(dims[3], embedding_dim, rank_ratio=0.5, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.norm(x)
        return self.head(x)


def EdgeFaceS(embedding_dim=512, **kwargs):
    return EdgeFace(embedding_dim=embedding_dim, dims=(48, 96, 160, 304), depths=(3, 3, 9, 3), **kwargs)
