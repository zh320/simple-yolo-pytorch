from .yolo import YOLO


def get_model(config):
    model_hub = {'yolo':YOLO}

    if config.model in model_hub.keys():
        model = model_hub[config.model](num_class=config.num_class, backbone_type=config.backbone_type, 
                            anchor_boxes=config.anchor_boxes)
    else:
        raise NotImplementedError(f"Unsupport model type: {config.model}")

    return model
