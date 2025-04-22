import torch
import torch.nn as nn
import torch.nn.functional as F


# Regular convolution with kernel size 3x3
def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                    padding=1, bias=bias)


# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, 
                    padding=0, bias=bias)


# Depth-wise seperable convolution with batchnorm and activation
class DSConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        super().__init__(
            DWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation, act_type, **kwargs),
            PWConvBNAct(in_channels, out_channels, act_type, **kwargs)
        )


# Depth-wise convolution -> batchnorm -> activation
class DWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                    dilation=1, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                        dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Point-wise convolution -> batchnorm -> activation
class PWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu', bias=True, **kwargs):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                    bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):    
            padding = (kernel_size - 1) // 2 * dilation

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Transposed /de- convolution -> batchnorm -> activation
class DeConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, 
                    padding=None, act_type='relu', **kwargs):
        super().__init__()
        if kernel_size is None:
            kernel_size = 2*scale_factor - 1
        if padding is None:    
            padding = (kernel_size - 1) // 2
        output_padding = scale_factor - 1
        self.up_conv = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels, out_channels, 
                                                        kernel_size=kernel_size, 
                                                        stride=scale_factor, padding=padding, 
                                                        output_padding=output_padding),
                                    nn.BatchNorm2d(out_channels),
                                    Activation(act_type, **kwargs)
                                    )

    def forward(self, x):
        return self.up_conv(x)


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super().__init__()
        activation_hub = {'relu': nn.ReLU,             'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU,    'prelu': nn.PReLU,
                          'celu': nn.CELU,              'elu': nn.ELU, 
                          'hardswish': nn.Hardswish,    'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU,              'glu': nn.GLU, 
                          'selu': nn.SELU,              'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid,        'softmax': nn.Softmax, 
                          'tanh': nn.Tanh,              'none': nn.Identity,
                        }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


def replace_act(module, tgt_act_type, src_act=nn.ReLU):
    for name, child_module in module.named_children():
        if isinstance(child_module, src_act):
            setattr(module, name, Activation(tgt_act_type))
        replace_act(child_module, tgt_act_type, src_act)


class SPP(nn.Module):
    def __init__(self, in_channel, out_channel, act_type, pool_sizes=[5,9,13]):
        super().__init__()
        self.conv1 = ConvBNAct(in_channel, in_channel//2, 1, act_type=act_type)
        self.pools = nn.ModuleList([nn.MaxPool2d(p_size, 1, p_size//2) for p_size in pool_sizes])
        self.conv2 = ConvBNAct(in_channel*2, out_channel, 1, act_type=act_type)

    def forward(self, x):
        x = self.conv1(x)

        x_pools = [x]
        for pool in self.pools:
            x_pools.append(pool(x))

        x = torch.cat(x_pools, dim=1)
        x = self.conv2(x)

        return x


class FPN(nn.Module):
    def __init__(self, channel, act_type):
        super().__init__()
        self.up1 = ConvBNAct(channel, channel//2, 1, act_type=act_type)
        self.up2 = nn.Sequential(
                            ConvBNAct(channel, channel//2, 3, act_type=act_type),
                            ConvBNAct(channel//2, channel//2, 3, act_type=act_type),
                            ConvBNAct(channel//2, channel//4, 1, act_type=act_type),
                        )
        self.up3 = nn.Sequential(
                        ConvBNAct(channel//2, channel//4, 3, act_type=act_type),
                        ConvBNAct(channel//4, channel//4, 3, act_type=act_type),
                    )

    def forward(self, x1, x2, x3):
        # 1/32
        x1 = self.up1(x1)

        # 1/16
        size2 = x2.size()[2:]
        x = F.interpolate(x1, size2, mode='nearest')
        x = torch.cat([x, x2], dim=1)
        x2 = self.up2(x)

        # 1/8
        size3 = x3.size()[2:]
        x = F.interpolate(x2, size3, mode='nearest')
        x = torch.cat([x, x3], dim=1)
        x3 = self.up3(x)

        return x1, x2, x3


class PAN(nn.Module):
    def __init__(self, channel, act_type):
        super().__init__()
        self.up1 = ConvBNAct(channel, channel//2, 1, act_type=act_type)
        self.up2 = nn.Sequential(
                            ConvBNAct(channel, channel//2, 3, act_type=act_type),
                            ConvBNAct(channel//2, channel//2, 3, act_type=act_type),
                            ConvBNAct(channel//2, channel//4, 1, act_type=act_type),
                        )
        self.head1 = nn.Sequential(
                        ConvBNAct(channel//2, channel//4, 3, act_type=act_type),
                        ConvBNAct(channel//4, channel//4, 3, act_type=act_type),
                    )
        self.down1 = ConvBNAct(channel//4, channel//4, 3, 2, act_type=act_type)
        self.head2 = nn.Sequential(
                        ConvBNAct(channel//2, channel//2, 3, act_type=act_type),
                        ConvBNAct(channel//2, channel//2, 3, act_type=act_type)
                    )
        self.down2 = ConvBNAct(channel//2, channel//2, 3, 2, act_type=act_type)
        self.head3 = nn.Sequential(
                        ConvBNAct(channel, channel, 3, act_type=act_type),
                        ConvBNAct(channel, channel, 3, act_type=act_type)
                    )

    def forward(self, x1, x2, x3):
        # 1/32
        x1 = self.up1(x1)

        # 1/16
        size2 = x2.size()[2:]
        x = F.interpolate(x1, size2, mode='nearest')
        x = torch.cat([x, x2], dim=1)
        x2 = self.up2(x)

        # 1/8
        size3 = x3.size()[2:]
        x = F.interpolate(x2, size3, mode='nearest')
        x = torch.cat([x, x3], dim=1)
        x_head1 = self.head1(x)

        # 1/16
        x = self.down1(x_head1)
        x = torch.cat([x, x2], dim=1)
        x_head2 = self.head2(x)

        # 1/32
        x = self.down2(x_head2)
        x = torch.cat([x, x1], dim=1)
        x_head3 = self.head3(x)

        return x_head1, x_head2, x_head3