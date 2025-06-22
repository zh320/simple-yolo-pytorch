import argparse

from datasets import list_available_datasets


def load_parser(config):
    args = get_parser()

    for k,v in vars(args).items():
        if v is not None:
            try:
                exec(f"config.{k} = v")
            except:
                raise RuntimeError(f'Unable to assign value to config.{k}')
    return config


def get_parser():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--task', type=str, default=None, choices = ['train', 'val', 'predict', 'debug'],
        help='choose which task you want to use')

    # Dataset
    dataset_list = list_available_datasets()
    parser.add_argument('--dataset', type=str, default=None, choices=dataset_list,
        help='choose which dataset you want to use')
    parser.add_argument('--dataroot', type=str, default=None, 
        help='path to your dataset')
    parser.add_argument('--num_class', type=int, default=None, 
        help='number of classes')
    parser.add_argument('--load_lbl_once', action='store_false', default=None, 
        help='load labels into memeroy once to avoid reading them multuple times')
    parser.add_argument('--min_label_area', type=int, default=None, 
        help='minimal area (pixel number) to keep a label')

    parser.add_argument('--train_voc2007', action='store_false', default=None, 
        help='whether to train VOC 2007 or not (default: True)')
    parser.add_argument('--train_voc2012', action='store_false', default=None, 
        help='whether to train VOC 2012 or not (default: True)')

    # Model
    parser.add_argument('--model', type=str, default=None, 
        choices=['yolo',],
        help='choose which model you want to use')
    parser.add_argument('--backbone_type', type=str, default=None, 
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'darknet'],
        help='choose which backbone you want to use')
    parser.add_argument('--channel_sparsity', type=float, default=None, 
        help="sparsity of channel for each layer, should be between (0, 1]")

    # Training
    parser.add_argument('--total_epoch', type=int, default=None, 
        help='number of total training epochs')
    parser.add_argument('--base_lr', type=float, default=None, 
        help='base learning rate for single GPU, total learning rate *= gpu number')
    parser.add_argument('--train_bs', type=int, default=None, 
        help='training batch size for single GPU, total batch size *= gpu number')

    # Validating
    parser.add_argument('--val_bs', type=int, default=None, 
        help='validating batch size for single GPU, total batch size *= gpu number')    
    parser.add_argument('--begin_val_epoch', type=int, default=None, 
        help='which epoch to start validating')    
    parser.add_argument('--val_interval', type=int, default=None, 
        help='epoch interval between two validations')
    parser.add_argument('--conf_thrs', type=float, default=None, 
        help='confidence threshold for validation')
    parser.add_argument('--max_nms_num', type=int, default=None, 
        help='max number of predicted objects which are sent to NMS')
    parser.add_argument('--val_iou', type=float, default=None, 
        help='IoU threshold for validation, use for NMS')

    # Testing
    parser.add_argument('--test_bs', type=int, default=None, 
        help='testing batch size (currently only support single GPU)')
    parser.add_argument('--test_data_folder', type=str, default=None, 
        help='path to your testing image folder')
    parser.add_argument('--test_conf_thrs', type=float, default=None, 
        help='confidence threshold for testing')
    parser.add_argument('--test_iou', type=float, default=None, 
        help='IoU threshold for testing, use for NMS')
    parser.add_argument('--class_map', type=dict, default=None,
        help='predefined dict to convert class names into labels')
    parser.add_argument('--color_map', type=list, default=None,
        help='predefined colormap for better visualization')

    # Debugging
    parser.add_argument('--num_debug_batch', type=int, default=None, 
        help='number of debugging batch size')
    parser.add_argument('--debug_dir', type=str, default=None, 
        help='path to save debugging results')

    # Loss
    parser.add_argument('--lambda_obj', type=float, default=None,
        help='coefficient for object/confidence loss')
    parser.add_argument('--lambda_coord', type=float, default=None,
        help='coefficient for coordinates/box loss')
    parser.add_argument('--lambda_noobj', type=float, default=None,
        help='coefficient for non-object loss')
    parser.add_argument('--lambda_scales', type=list, default=None,
        help='coefficients for different detection layers')
    parser.add_argument('--use_noobj_loss', action='store_true', default=None,
        help='whether to use non-object loss or not (default: False)')
    parser.add_argument('--label_assignment_method', type=str, default=None, 
        choices=['vanilla',],
        help='choose which label assignment method you want to use')
    parser.add_argument('--grid_sizes', type=list, default=None,
        help='sizes of grid for different detection layers')    
    parser.add_argument('--iou_loss_type', type=str, default=None, 
        choices=['iou', 'ciou', 'diou', 'giou', 'siou'],
        help='choose which iou method for coordinates/box loss you want to use')
    parser.add_argument('--focal_loss_gamma', type=float, default=None,
        help='coefficient of gamma for focal loss')
    parser.add_argument('--match_iou_thres', type=float, default=None,
        help='IoU threshold to determine whether a GT box match a given anchor or not')
    parser.add_argument('--filter_by_max_iou', action='store_false', default=None,
        help='whether to filter matches by max IoU or not when there are multiple matches per anchor (default: True)')
    parser.add_argument('--assign_conf_method', type=str, default=None, 
        choices=['iou', 'constant'],
        help='choose which confidence assignment method of GT for confidence/object loss you want to use')

    # Scheduler
    parser.add_argument('--lr_policy', type=str, default=None, 
        choices = ['cos_warmup', 'linear', 'step'],
        help='choose which learning rate policy you want to use')
    parser.add_argument('--warmup_epochs', type=int, default=None, 
        help='warmup epoch number for `cos_warmup` learning rate policy')

    # Optimizer
    parser.add_argument('--optimizer_type', type=str, default=None, 
        choices = ['sgd', 'adam', 'adamw'],
        help='choose which optimizer you want to use')
    parser.add_argument('--momentum', type=float, default=None, 
        help='momentum of SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=None, 
        help='weight decay rate of SGD optimizer')

    # Monitoring
    parser.add_argument('--save_ckpt', action='store_false', default=None,
        help='whether to save checkpoint or not (default: True)')
    parser.add_argument('--save_dir', type=str, default=None, 
        help='path to save checkpoints and training configurations etc.')
    parser.add_argument('--use_tb', action='store_false', default=None,
        help='whether to use tensorboard or not (default: True)')
    parser.add_argument('--tb_log_dir', type=str, default=None, 
        help='path to save tensorboard logs')
    parser.add_argument('--ckpt_name', type=str, default=None, 
        help='given name of the saved checkpoint, otherwise use `last` and `best`')

    # Training setting
    parser.add_argument('--amp_training', action='store_true', default=None,
        help='whether to use automatic mixed precision training or not (default: False)')
    parser.add_argument('--resume_training', action='store_false', default=None,
        help='whether to load training state from specific checkpoint or not if present (default: True)')
    parser.add_argument('--load_ckpt', action='store_false', default=None,
        help='whether to load given checkpoint or not if exist (default: True)')
    parser.add_argument('--load_ckpt_path', type=str, default=None, 
        help='path to load specific checkpoint, otherwise try to load `last.pth`')
    parser.add_argument('--base_workers', type=int, default=None, 
        help='number of workers for single GPU, total workers *= number of GPU')
    parser.add_argument('--random_seed', type=int, default=None, 
        help='random seed')
    parser.add_argument('--use_ema', action='store_true', default=None,
        help='whether to use exponetial moving average to update weights or not (default: False)')

    # YOLO setting
    parser.add_argument('--anchor_boxes', type=list, default=None, 
        help='predefined anchor boxes for YOLO')

    # Augmentation
    parser.add_argument('--img_size', type=list, default=None, 
        help='image size for training/validating/testing')
    parser.add_argument('--randscale', type=list, default=None, 
        help='scale limit for RandomScale augmentation')
    parser.add_argument('--perspective_range', type=list, default=None, 
        help='scale limit for Perspective augmentation')
    parser.add_argument('--perspective_p', type=float, default=None, 
        help='probability to perform Perspective')
    parser.add_argument('--rotate_limit', type=list, default=None, 
        help='rotate angle limit for Rotate augmentation')
    parser.add_argument('--rotate_p', type=float, default=None, 
        help='probability to perform Rotate')
    parser.add_argument('--brightness', type=float, default=None, 
        help='brightness limit for ColorJitter augmentation')
    parser.add_argument('--contrast', type=float, default=None, 
        help='contrast limit for ColorJitter augmentation')
    parser.add_argument('--saturation', type=float, default=None, 
        help='saturation limit for ColorJitter augmentation')
    parser.add_argument('--hue', type=float, default=None, 
        help='hue limit for ColorJitter augmentation')
    parser.add_argument('--h_flip', type=float, default=None, 
        help='probability to perform HorizontalFlip')
    parser.add_argument('--mosaic_p', type=float, default=None, 
        help='probability to perform Mosaic training')

    # DDP
    parser.add_argument('--synBN', action='store_true', default=None, 
        help='whether to use SyncBatchNorm or not if trained with DDP (default: False)')
    parser.add_argument('--local_rank', type=int, default=None, 
        help='used for DDP, DO NOT CHANGE')

    args = parser.parse_args()
    return args