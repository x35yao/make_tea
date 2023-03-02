import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class MAB(nn.Module):
    def __init__(self, embed_dim):
        super(MAB, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 8)
        self.rff = nn.Linear(embed_dim, embed_dim)

    def forward(self, Y, X):
        attn_output, _ = self.multihead_attn(Y, X, X)
        H = attn_output + Y
        H_output = self.rff(H)
        output = H_output + H
        return output

class SAB(nn.Module):
    def __init__(self, embed_dim):
        super(SAB, self).__init__()
        self.mab = MAB(embed_dim)

    def forward(self, X):
        output = self.mab(X, X)
        return output

class PoolMA(nn.Module):
    def __init__(self, embed_dim):
        super(PoolMA, self).__init__()
        self.mab = MAB(embed_dim)
        self.rff = nn.Linear(embed_dim, embed_dim)
        self.s_param = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.s_param)

    def forward(self, Z):
        z = self.rff(Z)
        attn_output = self.mab(self.s_param.repeat(1, Z.shape[1], 1), z)
        return attn_output

class TrajectoryModel(nn.Module):
    def __init__(self, embed_dim):
        super(SetTransformer, self).__init__()
        self.sab1 = SAB(embed_dim)



