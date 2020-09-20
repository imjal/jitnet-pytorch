# import some common libraries
import numpy as np
import os, json, cv2, random
from model import mask_rcnn_pred
from utils.metrics import meanIOU
from model.networks.jitnet import JITNet
from utils.visualize import visualize_masks
import utils.img_trans as img_trans
from model.losses import CrossEntropy2d
import torch
import torch.nn as nn
import time
import pdb

def train_model(opt, model, batch):
    # get mask-rcnn preds
    batch = batch.unsqueeze(0)
    outputs = model(batch)
    return outputs

def update_model(opt, loss_fn, optimizer, outputs, maskrcnn_outputs):
    maskrcnn_outputs = maskrcnn_outputs.to(opt.device)
    loss = loss_fn(outputs, maskrcnn_outputs)
    optimizer.zero_grad()
    loss.backward()
    del loss

def run_model(opt, model, batch, pre_process=True):
    # run model
    if pre_process:
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
    # load model weights here
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(opt.device)
    loss_fn = CrossEntropy2d(255)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    metric = meanIOU(opt)

    if opt.save_video:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      vid_pth = os.path.join(opt.save_dir, opt.exp_id + '_pred')
      out_pred = cv2.VideoWriter('{}.mp4'.format(vid_pth),fourcc,
        opt.framerate, (opt.input_w, opt.input_h))

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
                mask_rcnn_output = maskrcnn_model.get_predictions(batch)
                batch, gt_label = img_trans.transform(batch, mask_rcnn_output)
                while(update):
                    start = time.time()
                    output = train_model(opt, model, batch)
                    end_train = time.time()
                    # save the stuff every iteration
                    acc = metric.get_score(gt_label, output, u)
                    end_acc = time.time()
                    if u < umax and acc < a_thresh:
                        update_model(opt, loss_fn, optimizer, output, gt_label)
                    else:
                        update = False
                    end = time.time()
                    print(u, end_train - start, end_acc - end_train, end - end_acc, acc)
                    u+=1
                    del output
                if acc > a_thresh:
                    delta = min(delta_max, 2 * delta)
                else:
                    delta = max(delta_min, delta / 2)
                output = run_model(opt, model, batch, False) # run model with new weights
            else:
                output = run_model(opt, model, batch)
                # if opt.acc_collect and (iter_id % opt.acc_interval == 0):
                #     acc = metric.get_score(mask_rcnn_output, output, iter_id)
                #     print(acc)
        else:
            output = run_model(opt, model, batch)
            # acc = metric.get_score(mask_rcnn_output, output, iter_id)
            # print(acc)
            # self.accum_coco.store_metric_coco(iter_id, batch, output, opt, is_baseline=True)

        if opt.save_video:
            viz_pred = visualize_masks(output)
            out_pred.write(viz_pred)
    
        del output, batch
        iter_id+=1