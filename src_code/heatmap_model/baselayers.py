import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from heatmap_model.utils import *
import math

# @GuopengLI, 16:11:15 9th Feb. 2022 (auto)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class MLP(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states

    
class Subnet_traj(nn.Module):
    """
    Sub graph network for trajectories, containing back and forth GRU layers
        [batch, nb_agents, nb_vectors, in_channels] --> [batch, nb_agents, out_channels]
    """
    def __init__(self, length, in_channels, cnn_channels, out_channels, nb_CNN_layers):
        super(Subnet_traj, self).__init__()
        self.out_channels = out_channels
        self.cnn_ini = nn.Conv1d(in_channels, cnn_channels, 3, padding=1)
        self.cnn = nn.ModuleList()
        for i in range(nb_CNN_layers-1):
            self.cnn.append(nn.Conv1d(cnn_channels, cnn_channels, 3, padding=1))
        
        self.gru_backward = nn.GRU(cnn_channels, out_channels, num_layers=1, batch_first=True)
        self.gru_forward = nn.GRU(cnn_channels+out_channels, out_channels, num_layers=1, batch_first=True)
        self.pooling = nn.Linear(length, 1)#nn.MaxPool1d(length)
        
    def forward(self, x):
        h = x.contiguous().view(-1, x.size(-2), x.size(-1)) #[BxN, nb_vectors, in_channels]
        h = torch.transpose(h, 1, 2) #[BxN, in_channels, nb_vectors]
        h = self.cnn_ini(h)
        for layer_index, layer in enumerate(self.cnn):
            h = layer(h)
        h = torch.transpose(h, 1, 2) #[BxN, nb_vectors, cnn_channels]
        hr = torch.flip(h,[1])
        
        r1, _ = self.gru_backward(hr)
        r1 = torch.cat((h, torch.flip(r1, [1])), -1)
        
        h, _ = self.gru_forward(r1) #[BxN, nb_vectors, out_channels]
        h = torch.transpose(h, 1, 2) #[BxN, out_channels, nb_vectors,]
        h = self.pooling(h)
        out = h.contiguous().view(x.size(0), -1, self.out_channels) #[BxN, out_channels]
        
        return out
    
class MultiHeadSelfAttention(nn.Module):
    """
    Self-attention graph neural layer with multiple or single attention head for dynamic graphs
        [batch, nb_polylines, in_channels] --> [batch, nb_polylines, attention_size]           "max"/"average"
                                            or [batch, nb_polylines, attention_size*nb_heads]  "concatenate"
    """
    def __init__(self, in_channels, attention_size,
                 nb_heads,
                 aggregation_mode="cat",
                 use_decay=False,
                 scale=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.nb_heads = nb_heads
        self.attention_size = attention_size
        self.aggregation_mode = aggregation_mode
        self.in_channels = in_channels
        self.use_decay = use_decay
        self.q_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)
        self.k_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)
        self.v_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)
        self.attention_decay = nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)
        if scale:
            self.d = math.sqrt(self.attention_size)
        else:
            self.d = 1
                        
    def transpose_attention(self, x):
        z = x.size()[:-1] + (self.nb_heads,self.attention_size)
        x = x.view(*z)
        return x.permute(0, 2, 1, 3) #(B, nb_heads, N, attention_size)

    def forward(self, x, adj):
        q_ini = self.q_layer(x)
        k_ini = self.k_layer(x)
        v_ini = self.v_layer(x)
        
        q = self.transpose_attention(q_ini)
        k = self.transpose_attention(k_ini)
        v = self.transpose_attention(v_ini)
        mask = adj.repeat(self.nb_heads,1,1,1).transpose(1,0)
        scores = torch.matmul(q/self.d, k.transpose(-1, -2))
        attention_weights = nn.Softmax(dim=-1)(scores-1e5*(1-mask))
        
        if self.use_decay:
            v = torch.cat([v[:, 0:1, 0:1, :] * self.attention_decay, v[:, 0:1, 1:, :]], dim=2)
                
        c = torch.matmul(attention_weights, v)
        c = c.permute(0, 2, 1, 3).contiguous()
        
        if self.aggregation_mode == "cat":
            new_shape = c.size()[:-2] +(self.nb_heads*self.attention_size,)
            out = c.view(*new_shape)
        elif self.aggregation_mode == "max":
            out = nn.MaxPool2d((self.nb_heads, 1))(c)
            out = out.squeeze(-2)
        return out

class MultiHeadCrossAttention(nn.Module):
    '''
    cross attention from lanes to agents or from agents to lanes
    '''
    def __init__(self, c_q, c_v,
                 attention_size,
                 nb_heads,
                 aggregation_mode="cat",
                 use_decay=False,
                 scale=True):
        super(MultiHeadCrossAttention, self).__init__()
        self.nb_heads = nb_heads
        self.attention_size = attention_size
        self.aggregation_mode = aggregation_mode
        self.use_decay = use_decay
        self.c_q = c_q
        self.c_v = c_v
       
        self.q_layer = nn.Linear(self.c_q, self.attention_size*self.nb_heads)
        self.k_layer = nn.Linear(self.c_v, self.attention_size*self.nb_heads)
        self.v_layer = nn.Linear(self.c_v, self.attention_size*self.nb_heads)
            
        self.attention_decay = nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)
        if scale:
            self.d = math.sqrt(self.attention_size)
        else:
            self.d = 1
            
    def transpose_attention(self, x):
        z = x.size()[:-1] + (self.nb_heads,self.attention_size)
        x = x.view(*z)
        return x.permute(0, 2, 1, 3) #(B, nb_heads, N, attention_size)

    def forward(self, query, key, c_mask):
        q_ini = self.q_layer(query)
        k_ini = self.k_layer(key)
        v_ini = self.v_layer(key)
        
        q = self.transpose_attention(q_ini)
        k = self.transpose_attention(k_ini)
        v = self.transpose_attention(v_ini)
        
        scores = torch.matmul(q/self.d, k.transpose(-1, -2))
        scores = scores.permute(1,2,0,3)-1e5*(1-c_mask)
        attention_weights = nn.Softmax(dim=-1)(scores.permute(2,0,1,3))

        if self.use_decay:
            v = torch.cat([v[:, 0:1, 0:1, :] * self.attention_decay, v[:, 0:1, 1:, :]], dim=2)

        c = torch.matmul(attention_weights, v)
        c = c.permute(0, 2, 1, 3).contiguous()
        if self.aggregation_mode == "cat":
            new_shape = c.size()[:-2] +(self.nb_heads*self.attention_size, )
            h = c.view(*new_shape)
        if self.aggregation_mode == "max":
            h = nn.MaxPool2d((self.nb_heads, 1))(c)
            h = h.squeeze(-2)
        return h
    
class DecoderResCat(nn.Module):
    def __init__(self, in_features, hidden_size, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states
    
class EgoAssign(nn.Module):
    def __init__(self, hidden_size):
        super(EgoAssign, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([MLP(2, hidden_size // 2),
                                     MLP(hidden_size, hidden_size // 2),
                                     MLP(hidden_size, hidden_size)])

    def forward(self, hidden_states, hego):
        he = hego[:,:,:self.hidden_size//2].repeat(1,hidden_states.size(1),1)
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(torch.cat([hidden_states, he], dim=-1))

        return hidden_states
    
class ToCoordinateCrossAttention(nn.Module):
    def __init__(self, in_channels, attention_size,
                 nb_heads,
                 aggregation_mode="cat",
                 use_decay=False,
                 scale=True):
        super(ToCoordinateCrossAttention, self).__init__()
        self.nb_heads = nb_heads
        self.attention_size = attention_size
        self.aggregation_mode = aggregation_mode
        self.in_channels = in_channels
        self.use_decay = use_decay
        
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(2, in_channels))
        self.mlp.append(nn.LeakyReLU())
        self.mlp.append(nn.Linear(in_channels, in_channels))
        self.mlp.append(nn.LeakyReLU())
        #self.mlp.append(nn.Dropout(0.3))
       
        self.q_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)
        self.k_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)
        self.v_layer = nn.Linear(self.in_channels, self.attention_size*self.nb_heads)

        
        self.attention_decay = nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)
        if scale:
            self.d = math.sqrt(self.attention_size)
        else:
            self.d = 1
        
        #self.drop = nn.Dropout(0.3)
            
    def transpose_attention(self, x):
        z = x.size()[:-1] + (self.nb_heads,self.attention_size)
        x = x.view(*z)
        return x.permute(0, 2, 1, 3) #(B, nb_heads, N, attention_size)

    def forward(self, x, coordinates, c_mask):
        mask = c_mask           
        for layer_index, layer in enumerate(self.mlp):
            if layer_index == 0:
                embed = layer(coordinates)
            else:
                embed = layer(embed)
        
        q_ini = self.q_layer(embed)
        k_ini = self.k_layer(x)
        v_ini = self.v_layer(x)
        
        q = self.transpose_attention(q_ini)
        k = self.transpose_attention(k_ini)
        v = self.transpose_attention(v_ini)
        
        scores = torch.matmul(q/self.d, k.transpose(-1, -2))
        scores = scores.permute(1,2,0,3)-1e5*(1-mask)
        attention_weights = nn.Softmax(dim=-1)(scores.permute(2,0,1,3))

        if self.use_decay:
            v = torch.cat([v[:, 0:1, 0:1, :] * self.attention_decay, v[:, 0:1, 1:, :]], dim=2)
        
        c = torch.matmul(attention_weights, v)
        c = c.permute(0, 2, 1, 3).contiguous()
        if self.aggregation_mode == "cat":
            new_shape = c.size()[:-2] +(self.nb_heads*self.attention_size, )
            c = c.view(*new_shape)
        if self.aggregation_mode == "max":
            c = nn.MaxPool2d((self.nb_heads, 1))(c)
            c = c.squeeze(-2)

        return c
    
class SubGraph(nn.Module):
    def __init__(self, c_in, hidden_size, length, depth):
        super(SubGraph, self).__init__()
        self.hidden_size = hidden_size
        self.c_in = c_in
        self.depth = depth
        self.fc = MLP(c_in, hidden_size)
        self.fc2 = MLP(hidden_size)

        self.layers = nn.ModuleList([MultiHeadSelfAttention(hidden_size, hidden_size //2, 2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        
        adj_ = torch.ones(length, length).float()
        self.adj = nn.Parameter(adj_, requires_grad = False)

    def forward(self, x):
        h = x.reshape(-1, x.size(-2), x.size(-1))
        h = self.fc(h)
        h = self.fc2(h)
        
        A = self.adj.unsqueeze(0).repeat(h.size(0), 1, 1)

        for layer_index, layer in enumerate(self.layers):
            temp = h
            h = layer(h, A)
            h = F.relu(h)
            h = h + temp
            h = self.layers_2[layer_index](h)
        h = h.reshape(x.size(0), x.size(1), x.size(2), self.hidden_size)
        return torch.max(h, dim=2)[0]