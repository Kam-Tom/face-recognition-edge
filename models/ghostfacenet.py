import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        reduction = max(4, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, reduction, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(reduction, channels, 1)

    def forward(self, x):
        s = self.pool(x)
        s = self.act(self.conv1(s))
        s = F.hardsigmoid(self.conv2(s))
        return x * s


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, dw_kernel=3, stride=1, use_act=True):
        super().__init__()
        intrinsic = out_channels // 2

        self.primary_conv = nn.Conv2d(in_channels, intrinsic, kernel, stride, kernel // 2, bias=False)
        self.primary_bn = nn.BatchNorm2d(intrinsic)
        self.primary_act = nn.PReLU(intrinsic) if use_act else nn.Identity()

        self.cheap_conv = nn.Conv2d(intrinsic, intrinsic, dw_kernel, 1, dw_kernel // 2, groups=intrinsic, bias=False)
        self.cheap_bn = nn.BatchNorm2d(intrinsic)
        self.cheap_act = nn.PReLU(intrinsic) if use_act else nn.Identity()

    def forward(self, x):
        a = self.primary_act(self.primary_bn(self.primary_conv(x)))
        b = self.cheap_act(self.cheap_bn(self.cheap_conv(a)))
        return torch.cat([a, b], dim=1)


class GhostModuleMultiply(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, dw_kernel=3, stride=1, use_act=True):
        super().__init__()
        self.ghost = GhostModule(in_channels, out_channels, kernel, dw_kernel, stride, use_act)

        self.att_pool = nn.AvgPool2d(2, 2)
        self.att_conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, bias=False)
        self.att_bn1 = nn.BatchNorm2d(out_channels)
        self.att_dw1 = nn.Conv2d(out_channels, out_channels, (1, 5), 1, (0, 2), groups=out_channels, bias=False)
        self.att_bn2 = nn.BatchNorm2d(out_channels)
        self.att_dw2 = nn.Conv2d(out_channels, out_channels, (5, 1), 1, (2, 0), groups=out_channels, bias=False)
        self.att_bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        g = self.ghost(x)
        att = self.att_pool(x)
        att = self.att_bn1(self.att_conv(att))
        att = self.att_bn2(self.att_dw1(att))
        att = self.att_bn3(self.att_dw2(att))
        att = torch.sigmoid(att)
        att = F.interpolate(att, size=g.shape[2:], mode="bilinear", align_corners=False)
        return g * att


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel=3, stride=1, se_ratio=0.0, use_ghost_module_multiply=False):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.stride = stride
        self.use_se = se_ratio > 0

        if use_ghost_module_multiply:
            self.expand = GhostModuleMultiply(in_channels, hidden_channels, kernel=1, use_act=True)
        else:
            self.expand = GhostModule(in_channels, hidden_channels, kernel=1, use_act=True)

        if stride > 1:
            self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel, stride, kernel // 2,
                                     groups=hidden_channels, bias=False)
            self.dw_bn = nn.BatchNorm2d(hidden_channels)

        if self.use_se:
            self.se = SEModule(hidden_channels, se_ratio)

        self.project = GhostModule(hidden_channels, out_channels, kernel=1, use_act=False)

        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel, stride, kernel // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.expand(x)
        if self.stride > 1:
            out = self.dw_bn(self.dw_conv(out))
        if self.use_se:
            out = self.se(out)
        out = self.project(out)
        return out + self.shortcut(x)


class GhostFaceNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU(16)

        self.bottlenecks = nn.Sequential(
            GhostBottleneck(16, 20, 20, kernel=3, stride=1),
            GhostBottleneck(20, 32, 64, kernel=3, stride=2),
            GhostBottleneck(32, 32, 92, kernel=3, stride=1, use_ghost_module_multiply=True),
            GhostBottleneck(32, 52, 92, kernel=5, stride=2, se_ratio=0.25, use_ghost_module_multiply=True),
            GhostBottleneck(52, 52, 156, kernel=5, stride=1, se_ratio=0.25, use_ghost_module_multiply=True),
            GhostBottleneck(52, 104, 312, kernel=3, stride=2, use_ghost_module_multiply=True),
            GhostBottleneck(104, 104, 260, kernel=3, stride=1, use_ghost_module_multiply=True),
            GhostBottleneck(104, 104, 240, kernel=3, stride=1, use_ghost_module_multiply=True),
            GhostBottleneck(104, 104, 240, kernel=3, stride=1, use_ghost_module_multiply=True),
            GhostBottleneck(104, 144, 624, kernel=3, stride=1, se_ratio=0.25, use_ghost_module_multiply=True),
            GhostBottleneck(144, 144, 872, kernel=3, stride=1, se_ratio=0.25, use_ghost_module_multiply=True),
            GhostBottleneck(144, 208, 872, kernel=5, stride=2, se_ratio=0.25, use_ghost_module_multiply=True),
            GhostBottleneck(208, 208, 1248, kernel=5, stride=1, use_ghost_module_multiply=True),
            GhostBottleneck(208, 208, 1248, kernel=5, stride=1, se_ratio=0.25, use_ghost_module_multiply=True),
            GhostBottleneck(208, 208, 1248, kernel=5, stride=1, use_ghost_module_multiply=True),
            GhostBottleneck(208, 208, 1248, kernel=5, stride=1, se_ratio=0.25, use_ghost_module_multiply=True),
        )

        self.conv2 = nn.Conv2d(208, 1248, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(1248)
        self.prelu2 = nn.PReLU(1248)

        self.gdconv = nn.Conv2d(1248, 1248, 7, 1, 0, groups=1248, bias=False)
        self.bn3 = nn.BatchNorm2d(1248)

        self.linear = nn.Conv2d(1248, embedding_dim, 1, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.bottlenecks(x)
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.gdconv(x))
        x = self.bn4(self.linear(x))
        return x.flatten(1)
