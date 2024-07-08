from torchmetrics.detection.mean_ap import MeanAveragePrecision


def get_det_metrics(iou_thresholds=None):
    metrics = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', iou_thresholds=iou_thresholds, 
                                    rec_thresholds=None, max_detection_thresholds=None, class_metrics=False, 
                                    extended_summary=False, average='macro')
    return metrics
