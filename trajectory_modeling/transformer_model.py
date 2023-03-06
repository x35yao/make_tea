import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoderLayer
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model-1, 2) * (-math.log(10000.0) / d_model-1))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
#         print(self.pe[:,:x.size(0)].shape)
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


class TrajectoryModel(nn.Module):
    def __init__(self,  traj_dim: int, embed_dim: int, nhead: int, d_hid: int,
                 dropout: float = 0.1):
        super(TrajectoryModel, self).__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.sab1 = TransformerEncoderLayer(embed_dim, nhead, d_hid, dropout)
        self.sab2 = TransformerEncoderLayer(embed_dim, nhead, d_hid, dropout)
        self.sab3 = TransformerEncoderLayer(embed_dim, nhead, d_hid, dropout)
#         self.tf 
        self.lin0 = nn.Linear(traj_dim, embed_dim)
        self.lin4 = nn.Linear(embed_dim, traj_dim)
        
    def forward(self, x, mask=None):
        x = self.lin0(x) * math.sqrt(self.d_model)
        x = self.pos_embed(x)
        x = self.sab1(x)
        x = self.sab2(x)
        x = self.sab3(x)
        x = self.lin4(x)
        return x



