import torch
import torch.nn as nn

from models.net.layers import DecoderLayer
from models.module.embedding.positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p, num_layers):
        super().__init__()
    
        self.pos_enc = PositionalEncoding(dim_model, dropout_p, max_len=5000)

        self.layers = nn.ModuleList([
            DecoderLayer(dim_model, num_heads, dropout_p) for _ in range(num_layers)
        ])

    def forward(self, tgt, enc_src, tgt_mask=None, tgt_pad_mask=None, src_pad_mask=None):
        tgt = self.pos_enc(tgt)
        tgt = torch.permute(tgt, (1,0,2)) 
       
        for layer in self.layers:
            tgt = layer(tgt, enc_src, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)

        return tgt