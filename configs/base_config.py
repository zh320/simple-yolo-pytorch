class BaseConfig:
    def __init__(self,):
        # Task
        self.task = 'train' # train, val, predict, debug

        # Dataset
        self.dataset = None
        self.dataroot = None
        self.num_class = -1
        self.load_lbl_once = True

        # VOC
        self.train_voc2007 = True
        self.train_voc2012 = True

        # Model
        self.model = None
        self.backbone_type = None

        # Training
        self.total_epoch = 300
        self.base_lr = 0.01
        self.train_bs = 16          # For each GPU

        # Validating
        self.val_bs = 16            # For each GPU
        self.begin_val_epoch = 0    # Epoch to start validation
        self.val_interval = 1       # Epoch interval between validation
        self.conf_thrs = 0.001

        # Testing
        self.test_bs = 16
        self.test_data_folder = None
        self.test_conf_thrs = 0.2
        self.test_iou = 0.4
        self.class_map = None
        self.color_map = None

        # Debugging
        self.num_debug_batch = 1
        self.debug_dir = None

        # Loss
        self.lambda_obj = 1.
        self.lambda_coord = 1.
        self.lambda_noobj = 1.
        self.lambda_scales = [1., 1., 1.]
        self.use_noobj_loss = False
        self.label_assignment_method = 'all_grid'
        self.grid_sizes = None
        self.iou_loss_type = 'iou'
        self.focal_loss_gamma = 0.
        self.match_iou_thres = 0.1
        self.filter_by_max_iou = True

        # Scheduler
        self.lr_policy = 'cos_warmup'
        self.warmup_epochs = 3

        # Optimizer
        self.optimizer_type = 'sgd'
        self.momentum = 0.9         # For SGD
        self.weight_decay = 1e-4    # For SGD

        # Monitoring
        self.save_ckpt = True
        self.save_dir = 'save'
        self.use_tb = True          # tensorboard
        self.tb_log_dir = None
        self.ckpt_name = None
        self.logger_name = 'yolo_trainer'

        # Training setting
        self.amp_training = False
        self.resume_training = True
        self.load_ckpt = True
        self.load_ckpt_path = None
        self.base_workers = 8
        self.random_seed = 1
        self.use_ema = False

        # YOLO setting
        self.anchor_boxes = None

        # Augmentation
        self.img_size = [512, 512]      # W, H
        self.randscale = [-0.5, 1.0]
        self.perspective_range = [0.05, 0.1]
        self.perspective_p = 0.
        self.rotate_limit = [-90, 90]
        self.rotate_p = 0.
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.hue = 0.
        self.h_flip = 0.5
        self.mosaic_p = 1.0

        # DDP
        self.synBN = False

    def init_dependent_config(self):
        if self.load_ckpt_path is None and self.task in ['train', 'debug']:
            self.load_ckpt_path = f'{self.save_dir}/last.pth'

        if self.tb_log_dir is None:
            self.tb_log_dir = f'{self.save_dir}/tb_logs/'

        if self.anchor_boxes is None:
            # Anchor boxes clustered from COCO dataset
            self.anchor_boxes = [
                                # Small anchor boxes
                                [[10,13], [16,30], [33,23]],
                                # Medium anchor boxes
                                [[30,61], [62,45], [59,119]],
                                # Large anchor boxes
                                [[116,90], [156,198], [373,326]]
                                ]
        else:
            assert len(self.anchor_boxes) > 0

        if self.task == 'predict':
            if self.class_map is None:
                raise ValueError('You need to provide class map for `predict` task.\n')

            if self.color_map is None:
                import numpy as np
                self.color_map = np.random.randint(0, 255, (self.num_class, 3)).tolist()

        if self.task == 'debug':
            if self.debug_dir is None:
                self.debug_dir = f'{self.save_dir}/{self.label_assignment_method}_results'