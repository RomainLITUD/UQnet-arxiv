from heatmap_model.baselayers import *

class VectorEncoder(nn.Module):
    """
    Encoder that encodes maps marking and trajectories for interaction dataset
        output: if agg_mode='max': [batch, nb_agents+nb_markings, attention_size]
                if agg_mode='cat': [batch, nb_agents+nb_markings, attention_size*nb_heads]
    """
    def __init__(self, para):
        super(VectorEncoder, self).__init__()
        self.para = para
        self.hidden_size = para['encoder_attention_size']
        self.maps_encoder = SubGraph(10, self.hidden_size, 5,3)
        self.traj_encoder = SubGraph(8, self.hidden_size, 9,3)
        self.inference = self.para['inference']
        
        self.a2l = MultiHeadCrossAttention(self.hidden_size, self.hidden_size, self.hidden_size//2, 2)
        
        self.mapgraph = nn.ModuleList()
        for i in range(2):
            if i==0:
                self.mapgraph.append(MultiHeadSelfAttention(self.hidden_size, self.hidden_size//2, 2))
                self.mapgraph.append(LayerNorm(self.hidden_size))
            else:
                self.mapgraph.append(MultiHeadSelfAttention(self.hidden_size, self.hidden_size//2, 2))
                self.mapgraph.append(LayerNorm(self.hidden_size))
        
        self.globalgraph = nn.ModuleList()
        for i in range(3):
            if i==0:
                self.globalgraph.append(MultiHeadSelfAttention(self.hidden_size, self.hidden_size//2, 2))
                self.globalgraph.append(LayerNorm(self.hidden_size))
            else:
                self.globalgraph.append(MultiHeadSelfAttention(self.hidden_size, self.hidden_size//2, 2))
                self.globalgraph.append(LayerNorm(self.hidden_size))
        self.act = nn.ReLU()
      
    def forward(self, splines, trajectories, laneletfeature, adj, af, c_mask):
        lf = laneletfeature.unsqueeze(-2).repeat(1,1,5,1)
        maps = torch.cat((splines, lf),-1)       
        h1 = self.maps_encoder(maps)
        
        for layer_index, layer in enumerate(self.mapgraph):
            if layer_index%2==0:
                x = h1
                h1 = layer(h1, af)
                h1 = self.act(h1)
                h1 = h1+x
            else:
                h1 = layer(h1)
        
        h2 = self.traj_encoder(trajectories)
        
        hl = h1+self.a2l(h1, torch.cat((h1,h2[:,:1]), 1), c_mask[:,:56])
        ht = torch.cat((hl, h2), -2)
        h = ht
        for layer_index, layer in enumerate(self.globalgraph):
            if layer_index%2==0:
                x = h
                h = layer(h, adj)
                h = self.act(h)
                h = h+x
            else:
                h = layer(h)
                
        if self.inference:
            return h1, ht, h
        else:
            return h1, ht, h, torch.cat((h1, h2[:,:1]), 1)