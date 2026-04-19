import math
import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.bn(self.conv(x))


class FFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.pw1 = ConvBN(dim, hidden)
        self.act = nn.GELU()
        self.pw2 = ConvBN(hidden, dim, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, groups=groups)
        self.repconv = nn.Conv2d(in_channels, out_channels, kernel // 2, stride, padding // 2, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x) + self.repconv(x))


class StemLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=4, kernel=3):
        super().__init__()
        padding = 0 if (kernel % 2) == 0 else kernel // 2
        blocks = math.ceil(patch_size ** 0.5)
        dims = [in_channels] + [out_channels // 2 ** i for i in range(blocks - 1, -1, -1)]

        layers = []
        for i in range(blocks):
            layers.append(RepConv(dims[i], dims[i + 1], kernel=kernel, stride=2, padding=padding))
            if i < blocks - 1:
                layers.append(nn.GELU())
        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x)


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        padding = 0 if (kernel % 2) == 0 else kernel // 2
        self.dw = RepConv(in_channels, in_channels, kernel=kernel, stride=2, padding=padding, groups=in_channels)
        self.pw = ConvBN(in_channels, out_channels)
        self.ffn = FFN(out_channels, out_channels * 2)

    def forward(self, x):
        x = self.pw(self.dw(x))
        return x + self.ffn(x)


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(1e-5 * torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        return self.beta + self.alpha * x


class MHLA(nn.Module):
    def __init__(self, dim, resolution):
        super().__init__()
        self.dim = dim
        self.res = resolution ** 2
        self.n_head = 4
        self.norm = Affine(dim)
        self.lin = nn.ModuleList([nn.Linear(self.res, self.res) for _ in range(self.n_head)])
        self.ls = nn.Parameter(1e-5 * torch.ones(dim, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x.reshape(B, self.dim, self.res))
        chunks = list(x.chunk(self.n_head, dim=1))
        for i in range(self.n_head):
            chunks[i] = self.lin[i](chunks[i])
        x = torch.cat(chunks, dim=1)
        return self.ls * x.reshape(B, C, H, W)


class RepMixBlock(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.token_mixer = RepConv(dim, dim, kernel=3, stride=1, padding=1, groups=dim)
        self.ffn = FFN(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.token_mixer(x)
        return x + self.ffn(x)


class MHLABlock(nn.Module):
    def __init__(self, dim, mlp_ratio, resolution):
        super().__init__()
        self.rep = RepConv(dim, dim, kernel=3, stride=1, padding=1, groups=dim)
        self.attn = MHLA(dim, resolution)
        self.ffn = FFN(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.rep(x)
        x = x + self.attn(x)
        return x + self.ffn(x)


class Stage(nn.Module):
    def __init__(self, dim, depth, mlp_ratio, resolution, mixer_type):
        super().__init__()
        if mixer_type == "repmix":
            blocks = [RepMixBlock(dim, mlp_ratio) for _ in range(depth)]
        elif mixer_type == "mhla":
            blocks = [MHLABlock(dim, mlp_ratio, resolution) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class FaceLiVTv2(nn.Module):
    def __init__(self, embedding_dim=512, img_size=112, patch_size=4,
                 dims=(48, 96, 192, 320), depths=(3, 3, 9, 3),
                 mixer_types=("repmix", "repmix", "mhla", "mhla"),
                 mlp_ratio=2, merge_kernel=3, final_feature_dim=1284):
        super().__init__()
        self.num_stage = len(depths)

        patch_embedds = [StemLayer(3, dims[0], patch_size=patch_size)]
        stages = []
        img_res = img_size // patch_size

        for i in range(self.num_stage):
            stages.append(Stage(dims[i], depths[i], mlp_ratio, img_res, mixer_types[i]))
            if i < self.num_stage - 1:
                patch_embedds.append(PatchMerging(dims[i], dims[i + 1], kernel=merge_kernel))
                img_res = math.ceil(img_res / 2)

        self.patch_embedds = nn.ModuleList(patch_embedds)
        self.stages = nn.ModuleList(stages)

        self.pre_head = nn.Sequential(
            ConvBN(dims[-1], final_feature_dim),
            ConvBN(final_feature_dim, final_feature_dim, kernel=4, groups=final_feature_dim),
        )

        self.head_bn = nn.BatchNorm1d(final_feature_dim)
        self.head_fc = nn.Linear(final_feature_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.num_stage):
            x = self.patch_embedds[i](x)
            x = self.stages[i](x)
        x = self.pre_head(x).flatten(1)
        return self.head_fc(self.head_bn(x))


def FaceLiVTv2S(embedding_dim=512, **kwargs):
    return FaceLiVTv2(
        embedding_dim=embedding_dim,
        dims=(48, 96, 192, 320),
        depths=(3, 3, 9, 3),
        mixer_types=("repmix", "repmix", "mhla", "mhla"),
        final_feature_dim=1284,
        **kwargs,
    )
