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
        print("inputs.shape: ", inputs.shape)
        targets = targets.view(num, -1)
        print("targets.shape: ", targets.shape)
        intersection = (inputs * targets).sum(1)
        print("intersection.shape: ", intersection.shape)
        dice = (2. * intersection + self.smooth) / (inputs.sum(1) + targets.sum(1) + self.smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target):
        pred = self.sigmoid(pred)
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        dice = 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))
        return dice