import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from heatmap_model.utils import *
from heatmap_model.baselayers import *

import math

class VectorDecoder(nn.Module):
    def __init__(self, para, drivable=False):
        super(VectorDecoder, self).__init__()
        self.para = para
        self.drivable = drivable
        self.prob_mode = para['prob_mode']
        self.hidden_size = para['encoder_attention_size']
        
        self.lanescore = LanescoreModule(self.hidden_size)
        self.ego2coor = EgoAssign(self.hidden_size)
        self.l2c = MultiHeadCrossAttention(self.hidden_size,self.hidden_size,self.hidden_size//2, 2)
        self.l2c2 = MultiHeadCrossAttention(self.hidden_size,self.hidden_size,self.hidden_size//2, 2)
        c_h = self.hidden_size
        self.convert = DecoderResCat(4*c_h, c_h, 1)

    def forward(self, hlane, hmid, hinteraction, coordinates, c_mask, masker):
        log_lanescore = self.lanescore(hlane, hmid, hinteraction[:,55:56], c_mask)
        lanescore = torch.exp(log_lanescore)
        _, sorted_indices = torch.topk(lanescore,55)
        
        lanescore_mask = torch.zeros_like(c_mask[:,:55])
        for i in range(hlane.size(0)):
            prob = 0.0
            for idx, each in enumerate(lanescore[i][sorted_indices[i]]):
                prob += each
                if prob > 0.95:
                    lanescore_mask[i][sorted_indices[i][:idx + 1]] = 1
                    break
        coords = coordinates.unsqueeze(0).repeat(hlane.size(0), 1, 1)
        if self.drivable:
            maskerc = masker.clone()
            maskerc[maskerc==1]=100
            maskerc[maskerc==0]=-9999
            coords = torch.minimum(coords, maskerc.view(hlane.size(0), -1).unsqueeze(-1).repeat(1,1,2))
        position1 = self.ego2coor(coords, hinteraction[:,55:56])
        position2 = self.l2c(position1, hmid, c_mask)
        position3 = self.l2c2(position1, hlane, lanescore_mask)
        li = torch.cat((hinteraction[:,55:56].repeat(1,position1.size(1), 1), position1,position2,position3),-1)
        
        heatmap = self.convert(li).squeeze()
        if self.prob_mode == 'nll':
            heatmap = nn.LogSoftmax(-1)(heatmap)
        return log_lanescore.float(), heatmap
    
class RegularizeDecoder(nn.Module):
    def __init__(self, para, drivable=False):
        super(RegularizeDecoder, self).__init__()
        self.drivable = drivable
        self.hidden_size = para['encoder_attention_size']
        self.prob_mode = para['prob_mode']
        
        self.map2ego = MultiHeadSelfAttention(self.hidden_size, self.hidden_size//2, 2)
        self.lnorm = LayerNorm(self.hidden_size)
        
        c_h = self.hidden_size
        self.convert = DecoderResCat(c_h, c_h, 1)
        self.act = nn.ReLU()
        
        self.heatmapdecoder = ToCoordinateCrossAttention(self.hidden_size, self.hidden_size//2, 2)

    def forward(self, hmae, coordinates, ar, c_mask, masker):
        x = hmae
        h = self.map2ego(hmae, ar)
        h = self.act(h)
        h = h+x
        h = self.lnorm(h)
        
        coords = coordinates.unsqueeze(0).repeat(hmae.size(0), 1, 1)
        if self.drivable:
            maskerc = masker.clone()
            maskerc[maskerc==1]=100
            maskerc[maskerc==0]=-9999
            coords = torch.minimum(coords, maskerc.view(hmae.size(0), -1).unsqueeze(-1).repeat(1,1,2))
        h = self.heatmapdecoder(h, coords, c_mask[:,:56])
        heatmap = self.convert(h).squeeze()
        if self.prob_mode == 'nll':
            heatmap = nn.LogSoftmax(-1)(heatmap)
        return heatmap

    
class LanescoreModule(nn.Module):
    def __init__(self, c_h):
        super(LanescoreModule, self).__init__()
        self.c_h = c_h
        self.lanescore_cross_attention = MultiHeadCrossAttention(c_h, c_h, c_h//2, 2)
        self.connect = DecoderResCat(3*c_h, c_h, 1)

    def forward(self, hlane, hmid, hego, c_mask):
        hlane_attention = self.lanescore_cross_attention(hlane, hmid, c_mask)
        hego_hidden = hego.repeat(1,55,1)
        hls = torch.cat((hego_hidden, hlane, hlane_attention),-1)
        hls = self.connect(hls).squeeze()
        log_lanescore = nn.LogSoftmax(-1)(hls-1e7*(1-c_mask[:,:55]))
        
        return log_lanescore