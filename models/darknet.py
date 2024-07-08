import torch.nn as nn

from .modules import ConvBNAct


class DarkNet(nn.Module):
    def __init__(self, arch_type='darknet53', classification=False, num_class=1000, act_type='relu'):
        super(DarkNet, self).__init__()
        arch_hub = {'darknet53':[1,2,8,8,4]}
        if arch_type not in arch_hub:
            raise NotImplementedError(f'Unsupported model type: {arch_type}\n')
        num_blocks = arch_hub[arch_type]
        self.classification = classification

        self.stage0 = nn.Sequential(
                                    ConvBNAct(3, 32, 3, act_type=act_type),
                                    ConvBNAct(32, 64, 3, 2, act_type=act_type)
                                    )
        self.stage1 = self._make_stage(num_blocks[0], 64, act_type)
        self.down1 = ConvBNAct(64, 128, 3, 2, act_type=act_type)
        self.stage2 = self._make_stage(num_blocks[1], 128, act_type)
        self.down2 = ConvBNAct(128, 256, 3, 2, act_type=act_type)
        self.stage3 = self._make_stage(num_blocks[2], 256, act_type)
        self.down3 = ConvBNAct(256, 512, 3, 2, act_type=act_type)
        self.stage4 = self._make_stage(num_blocks[3], 512, act_type)
        self.down4 = ConvBNAct(512, 1024, 3, 2, act_type=act_type)
        self.stage5 = self._make_stage(num_blocks[4], 1024, act_type)

        if classification:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(1024, num_class)

    def _make_stage(self, num_block, channels, act_type):
        layers = [ResidualBlock(channels, act_type) for _ in range(num_block)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.down1(x)
        x1 = self.stage2(x)
        x = self.down2(x1)
        x2 = self.stage3(x)
        x = self.down3(x2)
        x3 = self.stage4(x)
        x = self.down4(x3)
        x4 = self.stage5(x)

        if self.classification:
            x = self.pool(x4)
            x = torch.flatten(x, 1)
            x = self.classifier(x)

            return x
        else:
            return x1, x2, x3, x4


class ResidualBlock(nn.Module):
    def __init__(self, channels, act_type):
        super(ResidualBlock, self).__init__()
        hid_channels = channels // 2
        self.conv = nn.Sequential(
                                ConvBNAct(channels, hid_channels, 1, act_type=act_type),
                                ConvBNAct(hid_channels, channels, 3, act_type=act_type),
                                )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual

        return x
