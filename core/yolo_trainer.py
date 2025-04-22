"""
Create by:  zh320
Date:       2024/06/15
"""

import os
import torch
from tqdm import tqdm
from torch.cuda import amp
from torchvision.ops import nms, batched_nms

from .base_trainer import BaseTrainer
from utils import (get_det_metrics, sampler_set_epoch, xywh_to_xyxy)


class YOLOTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if config.task in ['train', 'val']:
            self.mAP = get_det_metrics().to(self.device)

        if config.task == 'debug':
            from .loss import get_loss_fn
            self.loss_fn = get_loss_fn(config, self.device)

    def train_one_epoch(self, config):
        self.model.train()

        sampler_set_epoch(config, self.train_loader, self.cur_epoch) 

        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, (images, bboxes, classes) in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            images = images.to(self.device, dtype=torch.float32)
            bboxes = bboxes.to(self.device, dtype=torch.float32)    
            classes = classes.to(self.device, dtype=torch.float32)    

            self.optimizer.zero_grad()

            # Forward path
            with amp.autocast(enabled=config.amp_training):
                preds = self.model(images)
                loss, (conf_loss, iou_loss, class_loss) = self.loss_fn(preds, bboxes, classes)

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('train/loss', loss.detach(), self.train_itrs)
                self.writer.add_scalar('train/conf_loss', conf_loss, self.train_itrs)
                self.writer.add_scalar('train/iou_loss', iou_loss, self.train_itrs)
                self.writer.add_scalar('train/class_loss', class_loss, self.train_itrs)

            # Backward path
            self.scaler.scale(loss).backward()

            # # Clip the gradients
            # self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.ema_model.update(self.model, self.train_itrs)

            if self.main_rank:
                pbar.set_description(('%s'*5) % 
                                (f'Epoch:{self.cur_epoch}/{config.total_epoch}{" "*4}|',
                                f'Loss:{loss.item():4.4g}{" "*4}|',
                                f'Conf Loss:{conf_loss:4.4g}{" "*4}|',
                                f'IoU Loss:{iou_loss:4.4g}{" "*4}|',
                                f'Class Loss:{class_loss:4.4g}{" "*4}|',)
                                )
        return

    @torch.no_grad()
    def validate(self, config, val_best=False):
        empty_tensor = torch.tensor([]).to(self.device)
        pbar = tqdm(self.val_loader) if self.main_rank else self.val_loader
        for (images, bboxes, classes) in pbar:
            images = images.to(self.device, dtype=torch.float32)
            bboxes = bboxes.to(self.device, dtype=torch.float32)
            classes = classes.to(self.device, dtype=torch.long)

            preds = self.ema_model.ema(images, is_training=False)

            _, _, height, width = images.shape
            outputs, targets = [], []
            for i, pred in enumerate(preds):
                pred_conf = pred[:, 0]

                pred_boxes = xywh_to_xyxy(pred[:, 1:5])
                pred_boxes[:, 0::2].clamp_(0, width)
                pred_boxes[:, 1::2].clamp_(0, height)

                cls_logits, pred_cls = preds[i][:, 5:].max(dim=1)
                pred_conf *= cls_logits
                pred = torch.cat([pred_conf.unsqueeze(1), pred_boxes, pred_cls.unsqueeze(1)], dim=1)
                output = pred[pred_conf > config.conf_thrs]

                # NMS per image
                kept_indices = self.nms(output)

                if kept_indices is None:
                    outputs.append(dict(boxes=empty_tensor, scores=empty_tensor, labels=empty_tensor.long()))
                else:
                    outputs.append(dict(boxes=output[kept_indices][:, 1:5], scores=output[kept_indices][:, 0], 
                                        labels=output[kept_indices][:, 5].long()))

                if bboxes.shape[0]:
                    targets.append(dict(boxes=bboxes[bboxes[:, 0]==i][:, 1:], 
                                        labels=classes[classes[:, 0]==i][:, 1].long()))
                else:
                    targets.append(dict(boxes=empty_tensor, labels=empty_tensor.long()))

            self.mAP.update(outputs, targets)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        val_results = self.mAP.compute()

        map50 = val_results['map_50']
        map50_95 = val_results['map']
        score = map50

        if self.main_rank:
            if val_best:
                self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.')
            else:
                self.logger.info(f' Epoch{self.cur_epoch} score: {score:.4f}    | ' + 
                                 f'best val-score so far: {self.best_score:.4f}\n')

            if config.use_tb and self.cur_epoch < config.total_epoch:
                self.writer.add_scalar('val/mAP@IoU=0.5', map50.cpu(), self.cur_epoch+1)
                self.writer.add_scalar('val/mAP@IoU=0.5:0.95', map50_95.cpu(), self.cur_epoch+1)

        self.mAP.reset()

        return score.to(self.device)

    @classmethod
    def nms(cls, output, class_agnostic=False, max_nms_num=100, nms_iou=0.6):
        if class_agnostic:
            kept_indices = nms(boxes=output[:, 1:5], scores=output[:, 0], iou_threshold=nms_iou)
        else:
            kept_indices = batched_nms(boxes=output[:, 1:5], scores=output[:, 0], idxs=output[:, 5], iou_threshold=nms_iou)

        if not kept_indices.shape[0]:
            return None

        if kept_indices.shape[0] > max_nms_num:
            kept_indices = kept_indices[:max_nms_num]

        return kept_indices

    @torch.no_grad()
    def predict(self, config):
        if config.DDP:
            raise ValueError('Predict mode currently does not support DDP.')

        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype("tools/ARIAL.TTF", 14)

        self.logger.info('\nStart predicting...\n')

        self.model.eval()
        for (images, images_aug, img_names) in tqdm(self.test_loader):
            images_aug = images_aug.to(self.device, dtype=torch.float32)

            preds = self.model(images_aug, is_training=False)

            height, width = images_aug.shape[2:]
            resize_ratio = torch.tensor(images.shape[1:3]) / torch.tensor(images_aug.shape[2:])

            outputs = []
            for i, pred in enumerate(preds):
                image = Image.fromarray(images[i].numpy())
                save_path = os.path.join(config.save_dir, img_names[i])

                pred_conf = pred[:, 0]
                pred_boxes = xywh_to_xyxy(pred[:, 1:5])
                pred_boxes[:, 0::2].clamp_(0, width)
                pred_boxes[:, 1::2].clamp_(0, height)
                cls_logits, pred_cls = pred[:, 5:].max(dim=1)
                pred_conf *= cls_logits
                pred = torch.cat([pred_conf.unsqueeze(1), pred_boxes, pred_cls.unsqueeze(1)], dim=1)
                output = pred[pred_conf > config.test_conf_thrs]

                # NMS per image
                kept_indices = self.nms(output, nms_iou=config.test_iou)

                if kept_indices is not None:
                    output = output[kept_indices].cpu()
                    conf = output[:, 0]
                    bbox = output[:, 1:5]
                    cls = output[:, 5].long()

                    bbox[:, 0::2] *= resize_ratio[1]
                    bbox[:, 1::2] *= resize_ratio[0]
                    bbox = bbox.long()

                    draw = ImageDraw.Draw(image)

                    for j in range(conf.shape[0]):
                        text = f'{config.class_map[cls[j].item()]}-{conf[j]:.2f}'
                        draw.rectangle(bbox[j].tolist(), outline=tuple(config.color_map[cls[j]]), width=2)
                        draw.text(bbox[j,:2].tolist(), text, fill='white', font=font)

                image.save(save_path)

    @torch.no_grad()
    def debug(self, config):
        if config.DDP:
            raise ValueError('Debug mode currently does not support DDP.')

        from utils.plot import visualize_assignments

        self.logger.info(f'\nStart visualizing {config.label_assignment_method} label assignment results...\n')
        for cur_itrs, (images, bboxes, classes) in enumerate(tqdm(self.train_loader)):
            if cur_itrs >= config.num_debug_batch:
                break

            bboxes, classes = bboxes.float(), classes.float()

            assigned_labels = self.loss_fn.label_assignment(bboxes, classes, config.train_bs, len(config.anchor_boxes))

            visualize_assignments(assigned_labels=assigned_labels, bboxes=bboxes, anchor_boxes=config.anchor_boxes,
                                    batch_idx=cur_itrs, img_size=config.img_size, image_stride=(8, 16, 32), save_dir=config.debug_dir)

        self.logger.info('Debug finished.\n')