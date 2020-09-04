# import some common libraries
import numpy as np
import os, json, cv2, random
from model import mask_rcnn_pred
from metrics import meanIOU
from model.networks.jitnet import JITNet
import utils.image_trans as img_trans
from model.losses import CrossEntropy2d
import torch
import torch.nn as nn

def train_model(opt, model, batch):
    # get mask-rcnn preds
    batch = batch.unsqueeze(0)
    outputs = model(batch)
    return outputs, gt_seg

def update_model(opt, loss_fn, optimizer, outputs, maskrcnn_outputs):
    maskrcnn_outputs = maskrcnn_outputs.to(opt.device)
    loss = loss_fn(outputs, maskrcnn_outputs)
    optimizer.zero_grad()
    loss.backward()
    del loss

def run_model(opt, model, batch):
    # run model
    batch = img_trans.transform(batch)
    batch = batch.unsqueeze(0)
    outputs = model(batch)
    return outputs

def run_video(opt, data_iter):
    delta_max = opt.delta_max
    delta_min = opt.delta_min
    delta = delta_min
    umax = opt.umax
    a_thresh = opt.acc_thresh
    
    iter_id = 0
    maskrcnn_model = mask_rcnn_pred.MaskRCNNPred()
    model = JITNet(80).double()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(opt.device)
    loss_fn = CrossEntropy2d(255)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    metric = meanIOU(opt)

    while True:
        # data loading
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        if opt.adaptive:
            print("iteration: ", iter_id)
            if iter_id % delta == 0:
                u = 0
                update = True
                mask_rcnn_output = maskrcnn.get_predictions(batch)
                while(update):
                    output = train_model(opt, model, batch)
                    import pdb; pdb.set_trace()
                    # save the stuff every iteration
                    acc = metric.get_score(mask_rcnn_output, output, u)
                    if u < umax and acc < a_thresh:
                        update_model(opt, loss_fn, optimizer, output, mask_rcnn_output)
                    else:
                        update = False
                    print(u)
                    u+=1
                    del output
                if acc > a_thresh:
                    delta = min(delta_max, 2 * delta)
                else:
                    delta = max(delta_min, delta / 2)
                output = run_model(opt, model, batch) # run model with new weights
            else:
                output = run_model(opt, model, batch)
                if opt.acc_collect and (iter_id % opt.acc_interval == 0):
                    acc = metric.get_score(mask_rcnn_output, output, iter_id)
                    print(acc)
                del output
        else:
            output, model_time = run_model(opt, model, batch)
            if opt.acc_collect:
                acc = metric.get_score(mask_rcnn_output, output, iter_id)
                print(acc)
                # self.accum_coco.store_metric_coco(iter_id, batch, output, opt, is_baseline=True)

        # if opt.save_video:
        #     out_pred.write(pred)
        #     out_gt.write(gt)
    
        del output, batch
        iter_id+=1
        
        
    
    




