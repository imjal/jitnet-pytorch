import os
import logging
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class EncDecBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, encoder=True, resize=2, size=None):
        super(EncDecBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride)
        self.sep1 = nn.Conv2d(out_planes, out_planes, kernel_size=(1, 3), stride=1, padding=1)
        self.sep2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 1), stride=1, groups=out_planes)
        if size:
            self.up1 = nn.Upsample(size =(size[0], size[1]), mode='nearest')
        else:
            self.up1 = nn.Upsample(scale_factor=resize, mode='nearest')
        self.encoder = encoder

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        shortcut = self.conv2(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.sep1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.sep2(x)
        x = x + shortcut
        if not self.encoder:
             x = self.up1(x)
        return x
        

class JITNet(nn.Module):
    def __init__(self, num_classes):
        super(JITNet, self).__init__()
        self.num_classes = num_classes + 1 # add background class

        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(3, 8, stride=1)
        
        self.max2d = nn.MaxPool2d((1, 1), stride=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv2 = conv3x3(8, 8, stride=1)

        self.enc1 = EncDecBlock(8, 64, stride=2)
        self.enc2 = EncDecBlock(64, 64, stride=2)
        self.enc3 = EncDecBlock(64, 128, stride=2)

        self.dec3 = EncDecBlock(128, 64, encoder=False, size=(45, 80))
        self.dec2 = EncDecBlock(128, 32, encoder=False, resize=2)
        self.dec1 = EncDecBlock(96, 32, encoder=False, resize=4)

        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = conv3x3(32, 32, stride=1)
        self.conv4 = conv3x3(32, 32, stride=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1)

        # output num_classes for # of channels
        self.conv6 = nn.Conv2d(32, num_classes, kernel_size= (1,1), stride=1, bias=False)

    
    def forward(self, x):
        # conv 3x3 2 0 8
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.max2d(x)

        # conv 3x3 2 0 8
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.max2d(x)
        
        # 3 encoder blocks
        res1 = self.enc1(x)
        res2 = self.enc2(res1)
        x = self.enc3(res2)
        
        # 3 decoder blocks
        x = self.dec3(x)
        in2 = torch.cat((x, res2), dim=1)
        x = self.dec2(in2)
        in1 = torch.cat((x, res1), dim=1)
        x = self.dec1(in1)

        # process afterwards
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        # upsample one more time
        x = self.up1(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv5(x)
        
        # output logits
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv6(x)
        return x

class JITNetPred(nn.Module):
  def __init__(self, opt, model):
    super(JITNetPred, self).__init__()
    self.opt = opt
    self.model = model
    
  def forward(self, batch):
    batch = batch.to(opt.device)
    outputs = self.model(batch)
    return outputs