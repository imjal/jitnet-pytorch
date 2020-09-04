import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossEntropy2d(nn.Module):
  def __init__(self, ignore_label=255):
      super(CrossEntropy2d, self).__init__()
      self.ignore_label = ignore_label

  def forward(self, predict, target, weight=None):
      """
          Args:
              predict:(n, c, h, w)
              target:(n, h, w)
              weight (Tensor, optional): a manual rescaling weight given to each class.
                                          If given, has to be a Tensor of size "nclasses"
      """
      assert not target.requires_grad
      assert predict.dim() == 4
      assert target.dim() == 3
      assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
      assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
      assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
      n, c, h, w = predict.size()
      target = target.long()
      loss = F.cross_entropy(predict, target, ignore_index=self.ignore_label, reduction='none')
      if weight:
        weight = weight.float() * 5.0
        weight[weight == 0] = 1.0
      return torch.mean(loss)