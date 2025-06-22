import torch.nn as nn

from .modules import ConvBNAct


class DarkNet(nn.Module):
    def __init__(self, arch_type='darknet53', classification=False, base_channel=32, num_class=1000, 
                    act_type='relu', channel_sparsity=1.):
        super().__init__()
        assert channel_sparsity > 0
        ch = int(base_channel * channel_sparsity)

        arch_hub = {'darknet53':[1,2,8,8,4]}
        if arch_type not in arch_hub:
            raise NotImplementedError(f'Unsupported model type: {arch_type}\n')
        num_blocks = arch_hub[arch_type]
        self.classification = classification

        self.stage0 = nn.Sequential(
                                    ConvBNAct(3, ch, 3, act_type=act_type),
                                    ConvBNAct(ch, ch*2, 3, 2, act_type=act_type)
                                    )
        self.stage1 = self._make_stage(num_blocks[0], ch*2, act_type)
        self.down1 = ConvBNAct(ch*2, ch*4, 3, 2, act_type=act_type)
        self.stage2 = self._make_stage(num_blocks[1], ch*4, act_type)
        self.down2 = ConvBNAct(ch*4, ch*8, 3, 2, act_type=act_type)
        self.stage3 = self._make_stage(num_blocks[2], ch*8, act_type)
        self.down3 = ConvBNAct(ch*8, ch*16, 3, 2, act_type=act_type)
        self.stage4 = self._make_stage(num_blocks[3], ch*16, act_type)
        self.down4 = ConvBNAct(ch*16, ch*32, 3, 2, act_type=act_type)
        self.stage5 = self._make_stage(num_blocks[4], ch*32, act_type)

        if classification:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(ch*32, num_class)

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
        super().__init__()
        hid_channels = channels // 2
        self.conv = nn.Sequential(
                                ConvBNAct(channels, hid_channels, 1, act_type=act_type),
                                ConvBNAct(hid_channels, channels, 3, act_type=act_type),
                                )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x + residual

        return x