import math
import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    '''
    Positional Encoding using sine and cosine
    '''
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dim_model = dim_model
        self.dropout = nn.Dropout(dropout_p)

        # Encoding
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding):
        token_embedding = torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float32)) * token_embedding

        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])