import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn
from scipy.sparse import csr_matrix
from skimage.transform import rescale
from skimage.measure import block_reduce
from skimage.filters import gaussian

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class InteractionDataset(Dataset):
    """
    filename: a list of files or one filename of the .npz file
    stage: {"train", "val", "test"}
    """
    def __init__(self, filenames, stage, para, mode, filters=True):
        self.stage= stage
        self.para = para
        self.resolution = para['resolution']
        self.use_sem = para['use_sem']
        self.mode = mode
        self.filters = filters
        if stage == 'train':
            self.T = []
            self.M = []
            self.L = []
            self.N_agents = []
            self.N_splines = []
            self.Y = []
            self.Adj = []
            if mode=='lanescore':
                self.S = []
            for filename in filenames:
                data = np.load('./interaction_merge/'+filename+'.npz', allow_pickle=True)

                self.T.append(data['trajectory'])
                self.M.append(data['maps'])
                self.L.append(data['lanefeature'])
                self.N_agents.append(data['nbagents'])
                self.N_splines.append(data['nbsplines'])
                self.Adj.append(data['adjacency'])
                self.Y.append(data['intention'])
                if mode=='lanescore':
                    self.S.append(data['lanescore'])
            self.T = np.concatenate(self.T, axis=0)
            self.M = np.concatenate(self.M, axis=0)
            self.L = np.concatenate(self.L, axis=0)
            self.N_agents = np.concatenate(self.N_agents, axis=0)
            self.N_splines = np.concatenate(self.N_splines, axis=0)
            self.Y = np.concatenate(self.Y, axis=0)
            if mode=='lanescore':
                self.S = np.concatenate(self.S, axis=0)
            self.Adj = np.concatenate(self.Adj, 0)
            
            if self.use_sem:
                data_mask = np.load('./interaction_merge/sem_train.npz', allow_pickle=True)
                self.mask = data_mask['mask']
            else:
                data_mask = np.load('./interaction_merge/mask_train.npz', allow_pickle=True)
                self.mask = data_mask['mask']
        else:
            data = np.load('./interaction_merge/'+filenames[0]+'.npz', allow_pickle=True)
            self.T = data['trajectory']
            self.M = data['maps']
            self.L = data['lanefeature']
            self.N_agents = data['nbagents']
            self.N_splines = data['nbsplines']
            self.Adj = data['adjacency']

            if stage=='val':
                if filenames[0]=='val':
                    if self.use_sem:
                        data_mask = np.load('./interaction_merge/sem_val.npz', allow_pickle=True)
                    else:
                        data_mask = np.load('./interaction_merge/mask_val.npz', allow_pickle=True)
                    self.Y = data['intention']
                    if mode=='lanescore':
                        self.S = data['lanescore']
                else:
                    if self.use_sem:
                        data_mask = np.load('./interaction_merge/sem_valall.npz', allow_pickle=True)
                    else:
                        data_mask = np.load('./interaction_merge/mask_valall.npz', allow_pickle=True)
                    self.Y = data['intention']
                    if mode=='lanescore':
                        self.S = data['lanescore']
            if stage=='test':
                if self.use_sem:
                    data_mask = np.load('./interaction_merge/sem_test.npz', allow_pickle=True)
                else:
                    data_mask = np.load('./interaction_merge/mask_test.npz', allow_pickle=True)
            self.mask = data_mask['mask']
        
    def __len__(self):
        return len(self.N_agents)
    
    def __getitem__(self, index):
        traj = torch.tensor(self.T[index]).float().to(device)
        splines = torch.tensor(self.M[index]).float().to(device)
        lanefeature = torch.tensor(self.L[index].toarray()).float().to(device)
        nb_agents = self.N_agents[index]
        nb_splines = self.N_splines[index]
        
        if self.mode=='densetnt':
            adjacency = np.zeros((81, 81))

            cross = np.zeros(81)
            cross[:nb_splines] = 1
            cross[55:nb_agents] = 1
            
            adjacency[:nb_splines][...,:nb_splines] = 1
            adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
            adjacency[:nb_splines][...,55:55+nb_agents] = 1
            adjacency[55:55+nb_agents][...,:nb_splines] = 1
            adj = torch.Tensor(adjacency).int().to(device)
            c_mask = torch.Tensor(cross).int().to(device)

            masker = self.mask[index].toarray()#.reshape((46,87,3))
            if self.filters:
                filtered_masker = gaussian(masker, sigma=1.5)
                masker = np.where(masker, filtered_masker, masker)
            if self.resolution != 1:
                masker = rescale(masker, int(1/self.resolution))
            masker = torch.tensor(masker.copy()).float().to(device)
            
            if self.stage != "test":
                y = torch.tensor(self.Y[index]).float().to(device)
                return traj, splines, masker, lanefeature, adj, c_mask, y
            else:
                return traj, splines, masker, lanefeature, adj, c_mask
        
        if self.mode=='lanescore':
            a = self.Adj[index].toarray()
            af = a.copy()#+np.eye(55)
            af[af<0] = 0
            pad = np.zeros((55,55))
            pad[:nb_splines,:nb_splines]=np.eye(nb_splines)
            
            Af = np.linalg.matrix_power(af+pad+af.T, 4)
            Af[Af>0]=1
            
            A_f = torch.Tensor(Af).float().to(device)
            
            adjacency = np.zeros((81, 81))
            adjacency[:nb_splines][...,:nb_splines] = 1
            adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
            adjacency[:nb_splines][...,55:55+nb_agents] = 1 #optional
            adjacency[55:55+nb_agents][...,:nb_splines] = 1
            adj = torch.Tensor(adjacency).int().to(device)
            
            adjego = np.zeros((56, 56))
            adjego[:nb_splines][...,:nb_splines] = 1
            adjego[55,55] = 1
            adjego[55:56][...,:nb_splines] = 1
            A_r = torch.Tensor(adjego).int().to(device)
            
            c_mask = torch.Tensor(adjacency[:,0]).int().to(device)
            if self.stage!='test':
                ls = torch.tensor(self.S[index]).float().to(device)
            masker = self.mask[index].toarray()#.reshape(46, 87, 3)
            if self.filters:
                filtered_masker = gaussian(masker, sigma=1.5)
                masker = np.where(masker, filtered_masker, masker)
            if self.resolution != 1:
                masker = rescale(masker, int(1/self.resolution))
            if self.use_sem:
                masker = torch.tensor(masker.copy()).int().to(device)
            else:
                masker = torch.tensor(masker.copy()).float().to(device)

            if self.stage != "test":
                y = torch.tensor(self.Y[index]).float().to(device)
                return traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask, y, ls
            else:
                return traj, splines, masker, lanefeature, adj, A_f, A_r, c_mask
            
        if self.mode=='testmodel':    
            a = self.Adj[index].toarray()
            af = a.copy()#+np.eye(55)
            al = a.copy()#+np.eye(55)
            af[af<0] = 0
            al[al>0] = 0
            al[al<0] = 1

            adjacency = np.zeros((81, 81))
            cross = np.zeros(81)
            cross[:nb_splines] = 1
            cross[55:nb_agents] = 1
            pad = np.zeros((55,55))
            pad[:nb_splines,:nb_splines]=np.eye(nb_splines)

            adjacency[:nb_splines][...,:nb_splines] = 1#np.eye(nb_splines)
            adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
            adjacency[:nb_splines][...,55:55+nb_agents] = 1
            adjacency[55:55+nb_agents][...,:nb_splines] = 1

            adj = torch.Tensor(adjacency).int().to(device)
            c_mask = torch.Tensor(cross).int().to(device)
            
            Af = af+pad#np.linalg.matrix_power(af+pad, 2)
            Al = al+pad
            Af[Af>0]=1
            Al[Al>0]=1
            
            A_f = torch.Tensor(Af).float().to(device)
            A_l = torch.Tensor(Al).float().to(device)
            
            masker = self.mask[index].toarray()#.reshape((46,87,3))
            if self.filters:
                filtered_masker = gaussian(masker, sigma=1.5)
                masker = np.where(masker, filtered_masker, masker)
            if self.resolution != 1:
                masker = rescale(masker, int(1/self.resolution))
            masker = torch.tensor(masker.copy()).float().to(device)
            
            if self.stage != "test":
                y = torch.tensor(self.Y[index]).float().to(device)
                return traj, splines, masker, lanefeature, adj, A_f, A_l, c_mask, y
            else:
                return traj, splines, masker, lanefeature, adj, A_f, A_l, c_mask

class InteractionTraj(Dataset):
    def __init__(self, filename, stage):
        data = np.load(filename, allow_pickle=True)
        self.D = data['traj']
        self.A = self.D[:,-1]
        self.Y = self.D[:,:-1]
        
        if stage == 'train':
            self.T = []
            for i in range(1,5):
                data = np.load('./interaction_merge/train'+str(i)+'.npz', allow_pickle=True)

                self.T.append(data['trajectory'][:,1])
            self.T = np.concatenate(self.T, axis=0)
        else:
            if stage == 'test':
                data = np.load('./interaction_merge/test.npz', allow_pickle=True)
            else:
                data = np.load('./interaction_merge/val.npz', allow_pickle=True)
            self.T = data['trajectory'][:,1]
         
        assert len(self.A) == len(self.T)
        
    def __len__(self):
        return len(self.A)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.T[idx]).float().to(device)
        anchor = torch.tensor(self.A[idx]).float().to(device)
        y = torch.tensor((self.Y[idx]).reshape(-1)).float().to(device)
        
        return x, anchor, y
    
class InferenceTraj(Dataset):
    def __init__(self, d):
        self.D = d
        
    def __len__(self):
        return len(self.D)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.D[idx]).float().to(device)
        return x
