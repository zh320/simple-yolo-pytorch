"""
Create by:  zh320
Date:       2024/06/15
"""

import torch, math
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(config, device):
    criterion = YOLOLoss(num_class=config.num_class, img_size=config.img_size, anchor_boxes=config.anchor_boxes,
                        lambda_coord=config.lambda_coord, lambda_obj=config.lambda_obj, lambda_noobj=config.lambda_noobj, 
                        lambda_scales=config.lambda_scales, use_noobj_loss=config.use_noobj_loss, iou_method=config.iou_method,
                        label_assignment_method=config.label_assignment_method, grid_sizes=config.grid_sizes,
                        focal_loss_gamma=config.focal_loss_gamma)  

    return criterion


class YOLOLoss(nn.Module):
    def __init__(self, num_class, img_size, anchor_boxes, lambda_coord=0.05, lambda_obj=1.0, lambda_noobj=0.5, 
                    lambda_scales=[4.0, 1.0, 0.25], use_noobj_loss=False, label_assignment_method='vanilla', 
                    grid_sizes=None, iou_method='iou', focal_loss_gamma=0.):
        super(YOLOLoss, self).__init__()
        self.num_attrib = 5 + num_class
        self.img_size = torch.tensor(img_size)
        self.anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32)

        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_scales = lambda_scales
        self.use_noobj_loss = use_noobj_loss
        self.label_assignment_method = label_assignment_method
        if grid_sizes is None:
            grid_sizes = [(img_size[0]//2**(3+i), img_size[1]//2**(3+i)) for i in range(3)]
        assert len(lambda_scales) == len(grid_sizes)

        self.grid_sizes = grid_sizes

        self.bce_loss = FocalLoss(gamma=focal_loss_gamma)
        self.iou_loss_func = IoULoss(iou_method)

    def forward(self, predictions, bboxes, classes):
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        num_labels = len(bboxes)
        num_pred_layer = len(predictions)

        if self.anchor_boxes.device != device:
            self.anchor_boxes = self.anchor_boxes.to(device)

        if self.img_size.device != device:
            self.img_size = self.img_size.to(device)

        if num_labels:
            assigned_labels, assigned_anchor_indices = self.label_assignment(bboxes, classes, batch_size, num_pred_layer)
            assert len(predictions) == len(assigned_labels)

        total_loss = 0.
        for i in range(num_pred_layer):
            # Extract predicted confidence scores, box coordinates, and class probabilities
            predicted = predictions[i]
            pred_conf = predicted[..., 0]
            pred_xy = predicted[..., 1:3]
            pred_wh = predicted[..., 3:5]
            pred_class = predicted[..., 5:]

            if num_labels:
                # Extract assigned confidence scores, box coordinates, and class probabilities
                assigned = assigned_labels[i].to(device)
                assigned_conf = assigned[..., 0]
                assigned_coord = assigned[..., 1:5]
                assigned_class = assigned[..., 5:]

                # Extract assigned boxes per detection layer
                assigned_anchor_per_layer = assigned_anchor_indices[i].to(device)
            else:
                assigned_conf = torch.zeros_like(pred_conf, device=device)

            # Compute confidence loss (binary cross-entropy)
            conf_loss = self.bce_loss(pred_conf, assigned_conf)

            if num_labels:
                # Only consider positive examples for iou loss and class loss
                pos_mask = assigned_conf > 0

                # Compute IoU loss for bounding box coordinates (need to avoid in-place operation)
                pred_xy_decode = pred_xy.sigmoid()
                pred_wh_decode = torch.exp(pred_wh).clamp(max=1E3) * assigned_anchor_per_layer
                pred_coord = torch.cat([pred_xy_decode, pred_wh_decode], dim=-1)

                iou_loss = self.iou_loss_func(pred_coord[pos_mask], assigned_coord[pos_mask]).mean()

                # Compute class loss (binary cross-entropy)
                class_loss = self.bce_loss(pred_class[pos_mask], assigned_class[pos_mask])
            else:
                iou_loss, class_loss = 0., 0.

            # Loss for one detection layer
            loss = self.lambda_obj * conf_loss + self.lambda_coord * iou_loss + class_loss

            if self.use_noobj_loss:
                loss += self.lambda_noobj * (1 - assigned_conf) * conf_loss

            # Loss for all detection layers
            total_loss += self.lambda_scales[i] * loss

        return total_loss

    def label_assignment(self, bboxes, classes, batch_size, num_pred_layer):
        if self.label_assignment_method == 'vanilla':
            assigned_labels, assigned_anchor_indices = [], []
            for i in range(num_pred_layer):
                grid_h, grid_w = self.grid_sizes[i]  # Size of the feature map
                anchor_boxes_per_layer = self.anchor_boxes[i]

                # Create an empty tensor for assigned boxes
                assigned_labels_per_layer = torch.zeros((batch_size, len(anchor_boxes_per_layer), grid_h, grid_w, self.num_attrib), device=bboxes.device)  
                assigned_anchor_per_layer = torch.zeros((batch_size, len(anchor_boxes_per_layer), grid_h, grid_w, 2), device=bboxes.device)

                if len(bboxes):
                    batch_idx, x_center, y_center, width, height = bboxes.unbind(dim=1)
                    _, cls = classes.unbind(dim=1)
                    batch_idx = batch_idx.long()
                    cls = cls.long()

                    idx_w = (x_center * grid_w).long()
                    idx_h = (y_center * grid_h).long()

                    dx = x_center * grid_w - idx_w
                    dy = y_center * grid_h - idx_h
                    dw = width * grid_w
                    dh = height * grid_h

                    anchor_width, anchor_height = (anchor_boxes_per_layer / self.img_size).unbind(dim=1)

                    # Calculate intersection over union (IoU)
                    width = width.unsqueeze(1).repeat(1, len(anchor_boxes_per_layer))
                    height = height.unsqueeze(1).repeat(1, len(anchor_boxes_per_layer))

                    intersection_width = torch.minimum(width, anchor_width)
                    intersection_height = torch.minimum(height, anchor_height)
                    intersection_area = intersection_width * intersection_height

                    label_area = width * height
                    anchor_area = anchor_width * anchor_height
                    union_area = label_area + anchor_area - intersection_area

                    iou = intersection_area / union_area
                    max_k = iou.argmax(dim=1)

                    # Set confidence score to 1
                    assigned_labels_per_layer[batch_idx, max_k, idx_h, idx_w, 0] = 1
                    # Set box coordinates
                    assigned_labels_per_layer[batch_idx, max_k, idx_h, idx_w, 1:5] = torch.stack([dx, dy, dw, dh], dim=1)
                    # Set class (one-hot encoding)
                    assigned_labels_per_layer[batch_idx, max_k, idx_h, idx_w, 5 + cls] = 1

                    assigned_anchor_per_layer[batch_idx, max_k, idx_h, idx_w] = anchor_boxes_per_layer[max_k]

                assigned_labels.append(assigned_labels_per_layer)
                assigned_anchor_indices.append(assigned_anchor_per_layer)

            return assigned_labels, assigned_anchor_indices
        else:
            raise NotImplementedError(f'Unsupported box assignment method: {self.label_assignment_method}\n')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        bce_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss:
    def __init__(self, iou_method='iou'):
        iou_method_hub = ('iou', 'ciou', 'diou', 'giou', 'siou')
        if iou_method not in iou_method_hub:
            raise ValueError(f'Invalid IoU method. Supported methods are {", ".join(iou_method_hub)}.\n')

        self.iou_method = iou_method

    def __call__(self, pred_boxes, target_boxes):
        # Calculate intersection and union
        intersection = self.intersection_area(pred_boxes, target_boxes)
        union = self.union_area(pred_boxes, target_boxes)

        # Calculate IoU
        iou = intersection / union
        
        if self.iou_method in ['ciou', 'diou']:
            pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
            target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
            center_distance = torch.norm(pred_center - target_center, p=2, dim=1)
        
        if self.iou_method in ['ciou', 'siou']:
            aspect_ratio_term = (self.arctan(pred_boxes[..., 2] / pred_boxes[..., 3]) - self.arctan(target_boxes[..., 2] / target_boxes[..., 3])) ** 2 / (4 * math.pi ** 2)

        if self.iou_method == 'ciou':
            iou -= center_distance ** 2 / self.diagonal_length_squared(pred_boxes, target_boxes) - aspect_ratio_term
        
        elif self.iou_method == 'diou':
            iou -= center_distance ** 2 / self.diagonal_length_squared(pred_boxes, target_boxes)
        
        elif self.iou_method == 'giou':
            enclosing_area = self.enclosing_area(pred_boxes, target_boxes)
            iou_loss = iou - (enclosing_area - union) / enclosing_area
        
        elif self.iou_method == 'siou':
            iou -= aspect_ratio_term

        return 1 - iou

    def intersection_area(self, boxes1, boxes2):
        tl = torch.max(boxes1[..., :2], boxes2[..., :2])
        br = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = (br - tl).clamp(min=0)
        return wh[..., 0] * wh[..., 1]

    def union_area(self, boxes1, boxes2):
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        return area1 + area2 - self.intersection_area(boxes1, boxes2)

    def enclosing_area(self, boxes1, boxes2):
        tl = torch.min(boxes1[..., :2], boxes2[..., :2])
        br = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        return (br - tl).clamp(min=0).prod(dim=1)

    def diagonal_length_squared(self, boxes1, boxes2):
        return ((boxes1[..., :2] - boxes1[..., 2:]) ** 2).sum(dim=1) + ((boxes2[..., :2] - boxes2[..., 2:]) ** 2).sum(dim=1)

    def arctan(self, x):
        return torch.atan(x)
