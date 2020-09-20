# jitnet-pytorch

## Installation
My environment with versions is listed at `jitnet-environment.yml`. 

However, the main packages needed are:
```
detectron2
pytorch
torchvision
cv2
PIL
```

## Running JITNet
To run on a video:
`python demo.py --demo /path/to/video --adaptive --save_video`

Currently, I'm still training the COCO pretrained model, however you are free to use your own pretrained smaller model as a substitute for JITNet.