import torch as T
from torch.nn import *
import torch.nn.functional as F
import numpy as np


class Encoder(Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.bilstm = LSTM(dim*2, dim, bidirectional=True, dropout=0.2, num_layers=1)
    
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
        o_t2a = T.zeros_like(emb_a, dtype=T.float32)
        o_t1a = T.zeros_like(emb_a, dtype=T.float32)
        
        o_t2b = T.zeros_like(emb_a, dtype=T.float32)
        o_t1b = T.zeros_like(emb_a, dtype=T.float32)
        assert self.num_block > 0
        for i in range(self.num_block):
            inp_a = T.cat([emb_a, o_t1a + o_t2a], dim=-1)
            inp_b = T.cat([emb_b, o_t1b + o_t2b], dim=-1)
            o_a, o_b = self.blocks[i](inp_a, inp_b)
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
    
    def forward(self, inp_a, inp_b):
        encoded_a = self.encoder(inp_a)
        encoded_b = self.encoder(inp_b)
        aligned_a = self.aligner(encoded_a, encoded_b)
        fused_a = self.fuser(encoded_a, aligned_a)
        aligned_b = self.aligner(encoded_b, encoded_a)
        fused_b = self.fuser(encoded_b, aligned_b)
        return fused_a, fused_b
        
        
class Fuser(Module):
    def __init__(self, dim):
        super(Fuser, self).__init__()
        self.G1 = Linear(dim * 8, dim)
        self.G2 = Linear(dim * 8, dim)
        self.G3 = Linear(dim * 8, dim)
        self.G = Linear(dim * 3, dim)
        
    def _fuse(self, z, z_new):
        z1 = self.G1(T.cat([z, z_new],dim=-1))
        z2 = self.G2(T.cat([z, z-z_new],dim=-1))
        z3 = self.G3(T.cat([z, z*z_new],dim=-1))
        z_o = self.G(T.cat([z1,z2,z3],dim=-1))
        return z_o
    
    def forward(self, z, z_new):
        z = self._fuse(z, z_new)
        return z


class Aligner(Module):
    def __init__(self, dim):
        super(Aligner, self).__init__()
        self.ff = Linear(dim*4, dim*4)
        
    def forward(self, ix, iother):
        ex = self.ff(ix)
        eother = self.ff(iother)
        align_x = ex.bmm(eother.transpose(1,2))
        align_x = F.softmax(align_x, dim=-1)
        aligned = align_x.bmm(iother)
        return aligned
    
    
class Prediction(Module):
    def __init__(self, dim):
        super(Prediction, self).__init__()
        self.H = Linear(dim*4, 1)
        
    def forward(self, a, b):
        a = T.max(a, dim=1)[0]
        b = T.max(b, dim=1)[0]
        y = self.H(T.cat([a,b,T.abs(a-b),a*b],dim=-1))
        y = F.sigmoid(y)
        return y




if __name__ == '__main__':
    RE2 = RE2()
    dummy_input_a = T.ones(32,10,768, dtype=T.float32)
    dummy_input_b = T.ones(32,10,768, dtype=T.float32)
    y = RE2(dummy_input_a, dummy_input_b)
    print(y.size())
    