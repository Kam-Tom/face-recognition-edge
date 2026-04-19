import torch.nn as nn


class DepthWise(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, stride=1, residual=False):
        super().__init__()
        self.residual = residual

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.prelu1 = nn.PReLU(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)

        self.conv3 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.residual:
            return x + out
        return out


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 128, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.prelu1 = nn.PReLU(128)

        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, groups=64, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.prelu2 = nn.PReLU(128)

        self.bottlenecks = nn.Sequential(
            DepthWise(128, 128, hidden_channels=128, stride=2),
            DepthWise(128, 128, hidden_channels=128, stride=1, residual=True),
            DepthWise(128, 128, hidden_channels=128, stride=1, residual=True),
            DepthWise(128, 128, hidden_channels=128, stride=1, residual=True),
            DepthWise(128, 128, hidden_channels=128, stride=1, residual=True),
            DepthWise(128, 256, hidden_channels=256, stride=2),
            DepthWise(256, 256, hidden_channels=256, stride=1, residual=True),
            DepthWise(256, 256, hidden_channels=256, stride=1, residual=True),
            DepthWise(256, 256, hidden_channels=256, stride=1, residual=True),
            DepthWise(256, 256, hidden_channels=256, stride=1, residual=True),
            DepthWise(256, 256, hidden_channels=256, stride=1, residual=True),
            DepthWise(256, 256, hidden_channels=256, stride=1, residual=True),
            DepthWise(256, 256, hidden_channels=512, stride=2),
            DepthWise(256, 256, hidden_channels=256, stride=1, residual=True),
            DepthWise(256, 256, hidden_channels=256, stride=1, residual=True),
        )

        self.conv3 = nn.Conv2d(256, 512, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.prelu3 = nn.PReLU(512)

        self.gdconv = nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, embedding_dim, bias=False)
        self.bn5 = nn.BatchNorm1d(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.bottlenecks(x)
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.bn4(self.gdconv(x))
        x = x.flatten(1)
        x = self.fc(x)
        return self.bn5(x)
