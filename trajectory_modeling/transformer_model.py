import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoderLayer, Transformer
    
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
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)


class TFModelFull(nn.Module):
    def __init__(self,  traj_dim: int, embed_dim: int, nhead: int, layers: int,
                 dropout: float = 0.4):
        super(TFModelFull, self).__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.tf = Transformer(embed_dim, nhead, num_encoder_layers=layers, num_decoder_layers=layers,
                              dropout=dropout, batch_first=True, dtype=torch.float64)
        self.lin0 = nn.Linear(traj_dim, embed_dim, dtype=torch.float64)
        self.lin4 = nn.Linear(embed_dim, traj_dim, dtype=torch.float64)
        
    def forward(self, obj_seq, traj_seq):
        obj_emb = self.lin0(obj_seq) * math.sqrt(self.d_model)
        traj_emb = self.lin0(traj_seq) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        x = self.tf(obj_emb, traj_emb)
        x = self.lin4(x)
        return x

class TFModelLite(nn.Module):
    def __init__(self, task_dim:int, traj_dim: int, embed_dim: int, nhead: int,
                 layers: int, dropout: float = 0.2):
        super(TFModelLite, self).__init__()
        self.d_model = embed_dim
        self.pos_embed = PositionalEncoding(embed_dim, dropout)
        self.tf = Transformer(embed_dim, nhead, num_encoder_layers=layers, num_decoder_layers=layers, batch_first=True, dtype=torch.float64)
        self.lin0 = nn.Linear(task_dim, embed_dim, dtype=torch.float64)
        self.lin4 = nn.Linear(embed_dim, traj_dim, dtype=torch.float64)
        
    def forward(self, obj_seq, traj_seq):
        obj_emb = self.lin0(obj_seq) * math.sqrt(self.d_model)
        traj_emb = self.lin0(traj_seq) * math.sqrt(self.d_model)
        traj_emb = self.pos_embed(traj_emb)
        x = self.tf(obj_emb, traj_emb)
        x = self.lin4(x)
        return x
