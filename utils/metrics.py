import numpy as np
import pdb
from abc import ABC, abstractmethod
import copy

# TODO: find some fix for this to make it less disgusting
def ret_categories():
    return [{'supercategory': 'person', 'id': 1, 'name': 'person'}, 
    {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
    {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, 
    {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, 
    {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}, 
    {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]

def ret_categories_downsized():
    return [{'supercategory': 'person', 'id': 0, 'name': 'person'}, 
    {'supercategory': 'vehicle', 'id': 1, 'name': '4 wheeler'}, {'supercategory': 'vehicle', 'id': 2, 'name': '2 wheeler'}]


class AccumCOCO:
    def __init__(self):
        self.cocoDt = []
        self.cocoGt = []
        self.gt_counter = 0
        self.dt_counter = 0
        self.remap_coco2bdd = get_remap(coco2bdd_class_groups)

    def add_det_to_coco(self, iter_id, dets, is_gt=False, is_baseline= False):
        '''
        convert a CenterNet detection to coco image instance
        '''
        def remap(mp, cls):
            if cls in mp:
                return mp[cls]
            else:
                return -1
        for i in range(len(dets)):
            bbox = [dets[i][0], dets[i][1], dets[i][2]-dets[i][0], dets[i][3]- dets[i][1]]
            res = {
                "image_id": int(iter_id), 
                "category_id": int(dets[i][5]), 
                "bbox": bbox,
                "score": dets[i][4],
                "area": bbox[2] * bbox[3] # box area
            }
            if is_gt:
                res['iscrowd'] = 0
                res['ignore'] = 0
                res['id'] = self.gt_counter
                self.gt_counter+=1
                self.cocoGt += [res]
            elif is_baseline:
                # remapped = remap(self.remap_coco2bdd, detectron_classes[int(dets[i][5])])
                # if remapped < 0:
                #     continue
                # res["category_id"] = remapped
                res['id'] = self.dt_counter
                self.dt_counter+=1
                self.cocoDt += [res]
            else:
                res['id'] = self.dt_counter
                self.dt_counter+=1
                self.cocoDt += [res]

    def store_metric_coco(self, imgId, batch, output, opt, is_baseline=False):
      dets = ctdet_decode( output['hm'], output['wh'], reg=output['reg'], cat_spec_wh=opt.cat_spec_wh, K=opt.K)
      predictions = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
      predictions[:, :, :4] *= opt.down_ratio * opt.downsample
      dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
      dets_gt = copy.deepcopy(dets_gt)
      dets_gt[:, :, :4] *= opt.down_ratio * opt.downsample
      self.add_det_to_coco(imgId, predictions[0], is_baseline=is_baseline)
      self.add_det_to_coco(imgId, dets_gt[0], is_gt=True)
    
    def get_gt(self):
        return {'annotations': self.cocoGt, 'categories': ret_categories()}
    
    def get_dt(self):
        return self.cocoDt

class Metric(ABC):
    def __init__(self, opt):
        self.opt = opt
    
    @abstractmethod
    def get_score(self, batch, outputs, iter_num):
        pass

class meanIOU(Metric):
    def get_score(self, mask_rcnn_outputs, outputs, iter_id):
        def convert2d(arr): # convert a one-hot into a thresholded array
            max_arr = arr.max(axis=0)
            new_arr = arr.argmax(axis = 0) + 1
            new_arr[max_arr < 0.1] = 0
            return new_arr
        # add to conf matrix for each image
        pred = convert2d(outputs.detach().cpu().numpy()[0])
        gt = copy.deepcopy(mask_rcnn_outputs[0]).numpy().astype(np.int64)
        N = outputs.detach().cpu().numpy()[0].shape[0] + 1
        conf_matrix = np.bincount(N * pred.reshape(-1) + gt.reshape(-1), minlength=N ** 2).reshape(N, N)
        
        acc = np.full(N, np.nan, dtype=np.float)
        iou = np.full(N, np.nan, dtype=np.float)
        tp = conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(conf_matrix, axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix, axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        return miou

def get_metric(opt):
    if opt.acc_metric == 'mAP':
        metric = regmAP(opt, opt.center_thresh)
    elif opt.acc_metric == 'meanIOU':
        metric = meanIOU(opt)
    return metric