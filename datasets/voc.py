import os
import albumentations as AT

from .base_dataset import BaseDataset
from .dataset_registry import register_dataset


@register_dataset
class VOC(BaseDataset):
    def __init__(self, config, mode='train'):
        super().__init__(config, mode)
        if config.train_voc2007 + config.train_voc2012 == 0:
            raise RuntimeError('You should at least select one VOC dataset to train.\n')

        self.class_map = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4, 'bus':5, 'car':6, 'cat':7,
                        'chair':8, 'cow':9, 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                        'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

        if mode == 'train':
            imgset_name = 'trainval.txt'
            years = []
            if config.train_voc2007:
                years.append('VOC2007')
            if config.train_voc2012:
                years.append('VOC2012')
        else:   # val
            imgset_name = 'test.txt'
            years = ['VOC2007']

        for year in years:
            imgset_path = os.path.join(self.data_folder, year, 'ImageSets', 'Main', imgset_name)
            with open(imgset_path, 'r') as f:
                train_imgset = f.readlines()

            for line in train_imgset:
                img_name = f'{line.strip()}.jpg'
                lbl_name = f'{line.strip()}.xml'

                img_path = os.path.join(self.data_folder, year, 'JPEGImages', img_name)
                lbl_path = os.path.join(self.data_folder, year, 'Annotations', lbl_name)

                self.images.append(img_path)
                self.labels.append(lbl_path)

        if mode == 'train':
            self.transform = AT.Compose([
                AT.RandomScale(scale_limit=config.randscale),
                AT.Perspective(scale=config.perspective_range, p=config.perspective_p),
                AT.Rotate(limit=config.rotate_limit, p=config.rotate_p),
                AT.PadIfNeeded(min_height=config.img_size[1], min_width=config.img_size[0], border_mode=0, value=(0,0,0)),
                AT.RandomCrop(height=config.img_size[1], width=config.img_size[0]),
                AT.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation, hue=config.hue),
                AT.HorizontalFlip(p=config.h_flip),
                ], bbox_params=AT.BboxParams(format='albumentations', label_fields=['class_labels'], min_area=10, min_visibility=0.01)
            )

        self._check_dataset()