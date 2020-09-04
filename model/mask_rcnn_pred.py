# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pdb


class MaskRCNNPred:
    def __init__(self, model_name = 'mask_rcnn_R_50_FPN_3x', thresh=0.5, transform=None):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{model_name}.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{model_name}.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        self.transform = transform
    
    def masks_to_seg(self, instances):
        seg = np.zeros(instances.pred_masks.shape[1:], dtype=np.uint8)
        for i in range(len(instances.pred_masks)):
            seg[np.nonzero(instances.pred_masks[i].numpy())] = int(instances.pred_classes[i])
        return seg

    def get_predictions(self, im):
        outputs = self.predictor(im)
        instances = outputs['instances'].to('cpu')
        seg_mask = self.masks_to_seg(instances)
        boxes = instances.pred_boxes
        return (seg_mask, boxes)