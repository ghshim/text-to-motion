import torch
import torch.nn as nn

from models.net.layers import EncoderLayer
from models.module.embedding.positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p, num_layers):
        super().__init__()

        self.pos_enc = PositionalEncoding(dim_model, dropout_p, max_len=5000)

        self.layers = nn.ModuleList([
            EncoderLayer(dim_model, num_heads, dropout_p) for _ in range(num_layers)
        ])

    def forward(self, src, src_pad_mask=None):
        src = self.pos_enc(src)            
        src = torch.permute(src, (1,0,2))
        
        for layer in self.layers:
            src = layer(src, src_pad_mask=src_pad_mask)
        
        return src
