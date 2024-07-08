from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # VOC
        self.dataset = 'voc'
        self.data_root = '/path/to/your/dataset'
        self.num_class = 20
        self.train_voc2007 = True
        self.train_voc2012 = True

        # Model
        self.model = 'yolo'
        self.backbone_type = 'resnet18'

        # Loss
        self.lambda_coord = 0.05
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_scales = [4.0, 1.0, 0.25]
        self.use_noobj_loss = False
        self.iou_method = 'siou'
        self.focal_loss_gamma = 3.5

        # Testing (VOC)
        self.is_testing = False
        self.test_bs = 1
        self.test_data_folder = 'imgs'
        self.test_conf_thrs = 0.4
        class_map = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4, 'bus':5, 'car':6, 'cat':7,
                    'chair':8, 'cow':9, 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                    'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}
        self.class_map = {v:k for k,v in class_map.items()}

        # Training setting
        self.use_ema = True
