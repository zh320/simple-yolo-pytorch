"""
Create by:  zh320
Date:       2024/06/15
"""

import torch
import torch.nn as nn

from .backbone import ResNet
from .darknet import DarkNet
from .modules import SPP, PAN, conv1x1, replace_act


class YOLO(nn.Module):
    def __init__(self, num_class=1, backbone_type='resnet18', anchor_boxes=None, act_type='leakyrelu'):
        super(YOLO, self).__init__()
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            last_channel = 512 if backbone_type in ['resnet18', 'resnet34'] else 2048

            # Change activations of torchvision pretrained model
            if act_type != 'relu':
                replace_act(self.backbone, act_type, nn.ReLU)

        elif backbone_type == 'darknet':
            self.backbone = DarkNet(act_type=act_type)
            last_channel = 1024
        else:
            raise NotImplementedError()

        self.spp = SPP(last_channel, last_channel, act_type)

        self.pan = PAN(last_channel, act_type)

        self.det_head = YOLOHead(last_channel, num_class, anchor_boxes)

    def forward(self, x, is_training=True):
        _, x2, x3, x4 = self.backbone(x)

        x4 = self.spp(x4)

        res1, res2, res3 = self.pan(x4, x3, x2)

        x = self.det_head([res1, res2, res3], is_training=is_training)

        return x


class YOLOHead(nn.Module):
    def __init__(self, in_channel, num_class, anchor_boxes):
        super(YOLOHead, self).__init__()
        assert anchor_boxes is not None, 'Anchor boxes must be given.\n'

        self.anchor_boxes = torch.tensor(anchor_boxes)
        assert self.anchor_boxes.shape[0] * self.anchor_boxes.shape[1] != 0
        assert self.anchor_boxes.shape[2] == 2  # width, height

        self.num_anchor = self.anchor_boxes.shape[1]
        self.num_attrib = num_class + 5     # 5: conf, dx, dy, dw, dh
        out_channel = self.num_anchor * self.num_attrib
        self.downsample_rate = torch.tensor([8, 16, 32])

        self.heads = nn.ModuleList([conv1x1(in_channel//2**(2-i), out_channel) for i in range(3)])

    def forward(self, feats, is_training=True):
        device = feats[0].device

        out = []
        for i, head in enumerate(self.heads):
            feat = head(feats[i])
            batch_size, _, grid_h, grid_w = feat.size()
            feat = feat.view(batch_size, self.num_anchor, self.num_attrib, grid_h, grid_w)
            feat = feat.permute(0, 1, 3, 4, 2).contiguous()

            if is_training:
                out.append(feat)

            else: 
                # When not training, we need to decode the output using position shifts and predefined anchor boxes.
                # The output in the last dimension will be [conf, dx, dy, width, height, class1, class2, ...]
                x_shift = torch.arange(grid_w).view(1, 1, 1, grid_w).repeat(batch_size, self.num_anchor, grid_h, 1).to(device)
                y_shift = torch.arange(grid_h).view(1, 1, grid_h, 1).repeat(batch_size, self.num_anchor, 1, grid_w).to(device)

                base_anchors = ((self.anchor_boxes[i] * self.downsample_rate[i]).unsqueeze(0).unsqueeze(2).unsqueeze(3)) \
                                .repeat(1, 1, grid_h, grid_w, 1).to(device, dtype=torch.float32)

                # Decode box confidence using sigmoid
                feat[..., 0] = feat[..., 0].sigmoid()

                # Decode box center coords by adding position shifts
                feat[..., 1] = (feat[..., 1].sigmoid() + x_shift) * self.downsample_rate[i]
                feat[..., 2] = (feat[..., 2].sigmoid() + y_shift) * self.downsample_rate[i]

                # Decode box width and height by multiplying predefined anchor boxes
                feat[..., 3:5] = torch.exp(feat[..., 3:5]) * base_anchors

                # Decode class logits using sigmoid
                feat[..., 5:] = feat[..., 5:].sigmoid()

                # Reshape the output in order to perform NMS
                out.append(feat.view(batch_size, -1, self.num_attrib))

        out = out if is_training else torch.cat(out, dim=1)

        return out
