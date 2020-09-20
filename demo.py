from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pdb
from opts import opts
from run_jitnet import run_video
video_ext = ['mp4', 'mov', 'avi', 'mkv']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(opt.demo)
    opt.framerate = cam.get(cv2.CAP_PROP_FPS)
    opt.input_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    opt.input_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _frame_from_video(video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                yield frame
            else:
                break

    video_generator = _frame_from_video(cam)
    run_video(opt, video_generator)
    
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
