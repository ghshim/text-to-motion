import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
from typing import List, Optional, Tuple, Any


def build_pose_embed(learnable_pos_embed, embed_dim, total_num_patches=None, drop_rate=0.1):
    # Initialize the positional embeddings
    assert learnable_pos_embed in ["sinusoidal", "learnable"], f"Learnable positional embedding should be: [sinusoidal, learnable]. Got {learnable_pos_embed}."
    if learnable_pos_embed == "learnable":
        pos_embed = LearnableEmbedding(embed_dim, total_num_patches, drop_rate, batch_first=True)
    elif learnable_pos_embed == "sinusoidal":
        pos_embed = PositionalEncoding(embed_dim, drop_rate, batch_first=True)
    return pos_embed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.use_mask = False

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[B,T,P]

        if not batch_first:
            pe = pe.transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, mask=None):
        # not used in the final model
        if self.batch_first:
            len_ = x.shape[1]
            x = x + self.pe[:, :len_, :]
        else:
            len_ = x.shape[0]
            x = x + self.pe[:len_, :, :]
        return self.dropout(x)

    def init_weights(self):
        return

class LearnableEmbedding(nn.Module):
    def __init__(self, d_model, len_model, dropout=0.1, batch_first=False):
        super(LearnableEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, len_model, d_model))
        if not batch_first:
            self.pe = self.pe.tranpose(0,1)
        self.use_mask = False

    def forward(self, x, mask=None):
        # not used in the final model
        x = x + self.pe
        return self.dropout(x)

    def init_weights(self):
        trunc_normal_(self.pe, std=0.02)