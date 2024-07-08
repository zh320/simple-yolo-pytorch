import os, random, torch, json
import numpy as np


def xyxy_to_xywh(xyxy):
    if not isinstance(xyxy, torch.Tensor) and not isinstance(xyxy, np.ndarray):
        raise ValueError("Unsupported datatype. Expected torch.Tensor or np.ndarray.")

    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    width = x2 - x1
    height = y2 - y1
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    if isinstance(xyxy, torch.Tensor):
        xywh = torch.stack([x_center, y_center, width, height], dim=1)
    elif isinstance(xyxy, np.ndarray):
        xywh = np.stack([x_center, y_center, width, height], axis=1)

    return xywh


def xywh_to_xyxy(xywh):
    if not isinstance(xywh, torch.Tensor) and not isinstance(xywh, np.ndarray):
        raise ValueError("Unsupported datatype. Expected torch.Tensor or np.ndarray.")

    x_center, y_center, width, height = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    if isinstance(xywh, torch.Tensor):
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    elif isinstance(xywh, np.ndarray):
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

    return xyxy


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def get_writer(config, main_rank):
    if config.use_tb and main_rank:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.tb_log_dir)
    else:
        writer = None
    return writer
    
    
def get_logger(config, main_rank):
    if main_rank:
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")

        log_path = f'{config.save_dir}/{config.logger_name}.log'
        logger.add(log_path, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")
    else:
        logger = None
    return logger


def save_config(config):
    config_dict = vars(config)
    with open(f'{config.save_dir}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)


def log_config(config, logger):
    keys = ['dataset', 'num_class', 'model', 'backbone_type',
            'optimizer_type', 'lr_policy', 'total_epoch', 'train_bs', 'val_bs',  
            'train_num', 'val_num', 'gpu_num', 'num_workers', 'amp_training', 
            'DDP', 'use_ema']

    config_dict = vars(config)
    infos = f"\n\n\n{'#'*25} Config Informations {'#'*25}\n" 
    infos += '\n'.join('%s: %s' % (k, config_dict[k]) for k in keys)
    infos += f"\n{'#'*71}\n\n"
    logger.info(infos)
