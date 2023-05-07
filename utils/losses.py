import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        inputs = torch.sigmoid(inputs)
        num = targets.size(0)
        inputs = inputs.view(num, -1)
        targets = targets.view(num, -1)
        intersection = (inputs * targets).sum(1)

        dice = (2. * intersection) / (inputs.sum(1) + targets.sum(1))

        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice