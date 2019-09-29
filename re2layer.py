import torch as T
from torch.nn import *
import torch.nn.functional as F
import numpy as np


class Encoder(Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.bilstm = LSTM(dim, dim/2, bidirectional=True, dropout=0.2, num_layers=1)
    
    def forward(self, x):
        t_h, (_, _) = self.bilstm(x)
        o = T.cat([x, t_h], dim=-1)
        return o


class RE2(Module):
    def __init__(self, dim=768, num_block=2):
        super(RE2, self).__init__()
        self.num_block = num_block
        self.blocks = []
        self.predictor = Prediction(dim)
        self.aligner = Aligner(dim)
        for i in range(num_block):
            self.blocks.append(RE2Block(dim, self.aligner))
        
    def forward(self, emb_a, emb_b):
        o_t2a = T.zeros_like(emb_a)
        o_t1a = T.zeros_like(emb_a)
        
        o_t2b = T.zeros_like(emb_a)
        o_t1b = T.zeros_like(emb_a)
        assert self.num_block > 0
        for i in range(self.num_block):
            inp_a = T.cat([emb_a, o_t1a + o_t2a])
            inp_b = T.cat([emb_b, o_t1b + o_t2b])
            o_a = self.blocks[i](inp_a)
            o_b = self.blocks[i](inp_b)
            o_t2a = o_t1a
            o_t1a = o_a
            o_t2b = o_t1b
            o_t1b = o_b
        
        y = self.predictor(o_a, o_b)
        return y


class RE2Block(Module):
    def __init__(self, dim, aligner):
        super(RE2Block, self).__init__()
        self.encoder = Encoder(dim)
        self.fuser = Fuser(dim)
        self.aligner = aligner
    
    def forward(self, inp_a, encoded_b):
        encoded_a = self.encoder(inp_a)
        aligned_a = self.aligner(encoded_a, encoded_b)
        fused_a = self.fuser(encoded_a, aligned_a)
        return fused_a
        
        
class Fuser(Module):
    def __init__(self, dim):
        super(Fuser, self).__init__()
        self.G1 = Linear(dim * 2, dim)
        self.G2 = Linear(dim * 2, dim)
        self.G3 = Linear(dim * 2, dim)
        self.G = Linear(dim * 3, dim)
        
    def _fuse(self, z, z_new):
        z1 = self.G1(T.cat([z, z_new]))
        z2 = self.G2(T.cat([z, z-z_new]))
        z3 = self.G3(T.cat([z, z*z_new]))
        z_o = self.G(T.cat([z1,z2,z3]))
        return z_o
    
    def forward(self, z, z_new):
        z = self._fuse(z, z_new)
        return z


class Aligner(Module):
    def __init__(self, dim):
        super(Aligner, self).__init__()
        self.ff = Linear(dim, dim)
        
    def forward(self, ix, iother):
        ex = self.ff(ix)
        eother = self.ff(iother)
        align_x = ex.bmm(eother.transpose(1,2))
        align_x = F.softmax(align_x)
        aligned = align_x.bmm(iother)
        print(aligned.size())
        return aligned
    
    
class Prediction(Module):
    def __init__(self, dim):
        super(Prediction, self).__init__()
        self.H = Linear(dim*4, 1)
        
    def forward(self, a, b):
        a = T.max(a)
        b = T.max(b)
        y = self.H(T.cat([a,b,T.abs(a-b),a*b]))
        y = F.sigmoid(y)
        return y




