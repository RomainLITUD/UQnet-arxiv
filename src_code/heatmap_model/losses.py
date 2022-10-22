import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn

import math
    
class FocalLoss_interaction(nn.Module):
    def __init__(self, para, weight=None, size_average=True):
        self.para = para
        self.xmax = para['xmax']
        self.ymin = para['ymin']
        self.ymax = para['ymax']
        self.sigmax = para['sigmax']
        self.sigmay = para['sigmay']
        self.dx, self.dy = para['resolution'], para['resolution']
        
        lateral = torch.tensor([i+0.5 for i in range(int(-self.xmax/self.dx), 
                                                     int(self.xmax/self.dx))])*self.dx
        longitudinal = torch.tensor([i+0.5 for i in range(int(self.ymin/self.dy), 
                                                     int(self.ymax/self.dy))])*self.dy

        self.len_x = lateral.size(0)
        self.len_y = longitudinal.size(0)
        self.x = lateral.repeat(self.len_y, 1).transpose(1,0)
        self.y = longitudinal.repeat(self.len_x, 1)
        super(FocalLoss_interaction, self).__init__()
        
    def bce(self, yp, y):
        loss = y*torch.log(yp)+(1.-y)*torch.log(1.-yp)
        return -torch.sum(loss)
    
    def forward(self, inputs, targets, alpha=0.25, gamma=2., smooth=1):
        inputs = inputs.float()
        ref = torch.zeros_like(inputs)
        for i in range(ref.size(0)):
            #xi,yi = math.floor((targets[i,0].item()+self.xmax)/self.dx), math.floor((targets[i,1].item()-self.ymin)/self.dy)
            #xc = -self.xmax+self.dx/2+xi*self.dx
            #yc = self.ymin+self.dy/2+yi*self.dy
            xc = torch.ones_like(self.x)*targets[i,0].item()
            yc = torch.ones_like(self.y)*targets[i,1].item()
            ref[i] = torch.exp(-((self.x-xc)**2/self.sigmax**2)/2 - ((self.y-yc)**2/self.sigmay**2)/2)
        
        inputs = inputs.view(-1)
        ref = ref.view(-1)
        BCE = F.binary_cross_entropy_with_logits(inputs, ref.float(), reduction='sum')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
class OverAllLoss(nn.Module):
    def __init__(self, para):
        super(OverAllLoss, self).__init__()
        self.lanescore_loss = nn.NLLLoss(reduction='mean')
        self.heatmap_loss = FocalLoss_interaction(para)
        self.l1_loss = nn.L1Loss()
    def forward(self, inputs, targets, lmbd=5):
        l0 = self.heatmap_loss(inputs[2], targets[1])
        l1 = self.lanescore_loss(inputs[0], targets[0].to(torch.int64))
        lmain = self.heatmap_loss(inputs[1], targets[1])
        rescale1 = torch.sigmoid(inputs[1])
        rescale2 = torch.sigmoid(inputs[2])
        inner = torch.minimum(rescale1/torch.max(rescale1), rescale2/torch.max(rescale2))
        coefficient = self.l1_loss(inner, rescale1/torch.max(rescale1))*lmbd+1
        
        return 10*l1+l0+coefficient*lmain
    
class LanescoreLoss(nn.Module):
    def __init__(self):
        #self.loss = nn.BCEWithLogitsLoss()
        super(LanescoreLoss, self).__init__()
    
    def forward(self, inputs, targets, alpha=0.25, gamma=2., smooth=1):
        ref = torch.zeros_like(inputs)
        for i in range(ref.size(0)):
            ref[i][int(targets[i])] = 1
        
        inputs = inputs.view(-1)
        ref = ref.view(-1)
        BCE = F.binary_cross_entropy_with_logits(inputs, ref.float(), reduction='sum')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
        #ls_loss = self.loss(inputs, ref, reduction='sum')               
        return focal_loss
