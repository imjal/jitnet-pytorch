from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import torch

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--demo', default='', help='path to image/ image folders/ video or "webcam"')
    self.parser.add_argument('--gpus', default='0,1,2,3')
    self.parser.add_argument('--adaptive', action='store_true',
                             help='use jitnet online adaptation, otherwise default to running pretrained model')
    self.parser.add_argument('--delta_max', type=int, default=64, help='number of iterations to skip updating in stream')
    self.parser.add_argument('--delta_min', type=int, default=8, help='number of iterations to skip updating in video stream')
    self.parser.add_argument('--umax', type=int, default=10, help='max number of iterations to update')
    
    self.parser.add_argument('--acc_thresh', type=int, default=80, help='accuracy threshold on when to stop updating')
    self.parser.add_argument('--results_dir', default='results', help='results dir for all exp')
    self.parser.add_argument('--exp_id', default='default', help='experiment_id')

    self.parser.add_argument('--save_video', action='store_true', help='Save the output of JITNet')

  def init(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.results_dir = os.path.join(os.path.abspath(os.getcwd()), opt.results_dir)
    opt.save_dir = os.path.join(opt.results_dir, opt.exp_id)
    os.makedirs(opt.save_dir, exist_ok=True)
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.device = 'cuda:0'
    torch.backends.cudnn.enabled = False
    return opt
