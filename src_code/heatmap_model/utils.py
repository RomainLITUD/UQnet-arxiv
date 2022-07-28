import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn
import argparse
from scipy.sparse import csr_matrix
from skimage.transform import rescale
from skimage.measure import block_reduce
from numpy.lib.stride_tricks import as_strided
from skimage.feature import peak_local_max
from scipy import ndimage, misc
from scipy.signal import convolve2d
from skimage.filters import gaussian

import math
from torch.optim.optimizer import Optimizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mask_softmax(x, test=None):

    if test == None:
        return F.softmax(x, dim=-1)
    else:
        shape = x.shape
        if test.dim() == 1:
            test = torch.repeat_interleave(
                test, repeats=shape[1], dim=0)
        else:
            test = test.reshape(-1)
        x = x.reshape(-1, shape[-1])
        for i, j in enumerate(x):
            j[int(test[i]):] = -1e5
        return F.softmax(x.reshape(shape), dim=-1)
    
def Entropy(x, resolution, epsilon=1e-6):
    x = np.reshape(x,(len(x),-1))
    x = x/np.sum(x,axis=-1)[:,np.newaxis]/resolution/resolution
    y = np.where(x>epsilon, -x*np.log(x)*resolution*resolution, 0)
    return np.sum(y,axis=-1)

def KLDivergence(x,y, resolution, epsilon=1e-5):
    assert x.shape == y.shape
    x = np.reshape(x,(len(x),-1))
    y = np.reshape(y,(len(y),-1))
    z = np.where(((y>epsilon)&(x>epsilon)), x*np.log(x/y)*resolution*resolution, 0)
    return np.sum(z,axis=-1)

def ComputeUQ(H, resolution, epsilon=1e-8):
    Ht = np.transpose(H, (2,3,0,1))
    Ht = (Ht/np.sum(Ht, axis=(0,1))).transpose((2,3,0,1))/resolution/resolution
    H_avr = np.mean(Ht, 0)
    #aleatoric = Entropy(H_avr, resolution, epsilon)
    aleatoric = np.zeros((len(H), H.shape[1]))
    for i in range(len(H)):
        aleatoric[i] = Entropy(Ht[i],resolution, epsilon)
    aleatoric = np.mean(aleatoric,0)
    
    epistemic = np.zeros((len(H), len(aleatoric)))
    for i in range(len(H)):
        epistemic[i] = KLDivergence(Ht[i], H_avr, resolution, epsilon)
    epistemic = np.mean(epistemic,0)
    return H_avr, aleatoric, epistemic

def inference_model(model, filename, dataset_name, number, dataset, para, k=6, test=False, return_heatmap=True, mode='densetnt', batch=16):
    H = []
    Ua = []
    Ue = []
    Yp = []
    L = []
    scale = int(1/para['resolution'])
    data = np.load('./interaction_merge/'+dataset_name+'.npz', allow_pickle=True)
    if return_heatmap:
        Na = data['nbagents']
        Nm = data['nbsplines']
        T = data['trajectory']
        M = data['maps']
    if not test:
        Y = data['intention']
        
    nb = len(dataset)
    cut = list(range(0, nb, 400*batch)) + [nb]
    
    for i in range(len(cut)-1):
        ind = list(range(cut[i], cut[i+1]))
        testset = torch.utils.data.Subset(dataset, ind)
        loader = DataLoader(testset, batch_size=batch, shuffle=False)
        
        Hp = []
        Lp = []
        for j in range(number):
            model.encoder.load_state_dict(torch.load(filename+'encoder'+ str(j) + '.pt'))
            model.decoder.load_state_dict(torch.load(filename+'decoder'+ str(j) + '.pt'))

            model.eval()
            Hi = []
            Li = []
            for k, data in enumerate(loader):
                print(i, j, k, end='\r')
                    
                if mode=='lanescore':
                    if not test:
                        traj, splines, masker, lanefeature, adj, af, ar, c_mask, y, ls = data
                    else:
                        traj, splines, masker, lanefeature, adj, af, ar, c_mask = data
                    lsp, heatmap = model(traj, splines, masker, lanefeature, adj, af, ar, c_mask)
                    Hi.append(heatmap.detach().to('cpu').numpy())
                    Li.append(lsp.detach().to('cpu').numpy())
            Hi = np.concatenate(Hi,0)
            Li = np.concatenate(Li,0)
            Hp.append(Hi)
            Lp.append(Li)
        Hp = np.stack(Hp, 0)
        Lp = np.stack(Lp, 0)
        hm, ua, ue = ComputeUQ(Hp, para['resolution'], epsilon=5e-4)
        print(hm.shape)
        yp = ModalSampling(hm, para, r=3, k=6)
        Ua.append(ua)
        Ue.append(ue)
        Yp.append(yp)
        if return_heatmap:
            H.append(hm)
            L.append(Lp.squeeze())
            
    Ua = np.concatenate(Ua, 0)
    Ue = np.concatenate(Ue, 0)
    Yp = np.concatenate(Yp, 0)
    if return_heatmap:
        H = np.concatenate(H, 0) 
    
    if test:
        if return_heatmap:
            return M, T, Nm, Na, Yp, Ua, Ue, H#, L
        else:
            return Yp, Ua, Ue
    else:
        if return_heatmap:
            return M, T, Nm, Na, Yp, Ua, Ue, H, Y
        else:
            return Yp, Ua, Ue, Y
    
def pool2d_np(A, kernel_size, stride=1, padding=0):

    A = np.pad(A, padding, mode='constant')
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)

    return A_w.max(axis=(2, 3))

def ModalSampling(H, paralist, r=2, k=6):
    # Hp = (N, H, W)
    dx, dy = paralist['resolution'], paralist['resolution']
    xmax, ymax = paralist['xmax'], paralist['ymax']
    ymin = paralist['ymin']
    Y = np.zeros((len(H), k, 2))
    for i in range(len(H)):
        print(i, end='\r')
        Hp = H[i].copy()
        y = np.zeros((k,2))
        xc, yc = np.unravel_index(Hp.argmax(), Hp.shape)
        xc=xc+r
        yc=yc+r
        pred = [-xmax+xc*dx+dx/2, ymin+yc*dy+dy/2]
        y[0] = np.array(pred)
        Hp[xc-r:xc+r+1,yc-r:yc+r+1] = 0.
        for j in range(1,k):
            Hr = pool2d_np(Hp, kernel_size=2*r+1, stride=1, padding=r)
            xc, yc = np.unravel_index(Hr.argmax(), Hr.shape)
            xc=xc+r
            yc=yc+r
            pred = [-xmax+xc*dx+dx/2, ymin+yc*dy+dy/2]
            y[j] = np.array(pred)
            Hp[xc-r:xc+r+1,yc-r:yc+r+1] = 0.
        Y[i] = y
    return Y

def ComputeError(Yp,Y, r=2, sh=6):
    assert sh <= Yp.shape[1]
    # Yp = [N,k,2], Y = [N,2]
    E = np.abs(Yp.transpose((1,0,2))-Y) #(k,N,2)
    FDE = np.min(np.sqrt(E[:sh,:,0]**2+E[:sh,:,1]**2), axis=0) #(N,)
    MR = np.where(FDE>r, np.ones_like(FDE), np.zeros_like(FDE))
    print("minFDE:", np.mean(FDE),"m")
    print("minMR:", np.mean(MR)*100,"%")
    return FDE, MR