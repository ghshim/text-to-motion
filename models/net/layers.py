import torch.nn as nn

from models.net.transformer.mha_utils import MultiheadAttentionStable


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p):
        super().__init__()
        self.self_attention = MultiheadAttentionStable(dim_model, num_heads, dropout=dropout_p)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Linear(dim_model * 4, dim_model),
            nn.Dropout(dropout_p)
        )
        
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, src_mask=None, src_pad_mask=None):
        # Self Attention
        src_, _ = self.self_attention(src, src, src, 
                                      attn_mask=src_mask,
                                      key_padding_mask=src_pad_mask)
        src = src + self.dropout(src_)
        src = self.norm1(src)
        
        # Feed Forward
        src_ = self.feed_forward(src)
        src = src + self.dropout(src_)
        src = self.norm2(src)
        
        return src


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p):
        super().__init__()
        self.masked_self_attention = MultiheadAttentionStable(dim_model, num_heads, dropout=dropout_p)
        self.encoder_decoder_attention = MultiheadAttentionStable(dim_model, num_heads, dropout=dropout_p)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Linear(dim_model * 4, dim_model),
            nn.Dropout(dropout_p)
        )

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, tgt, memory, tgt_mask=None, tgt_pad_mask=None, src_pad_mask=None):
        # Masked Self-Attention
        tgt2, _ = self.masked_self_attention(tgt, tgt, tgt, 
                                             attn_mask=tgt_mask,
                                             key_padding_mask=tgt_pad_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Encoder-Decoder Attention
        tgt2, _ = self.encoder_decoder_attention(tgt, memory, memory,
                                                 key_padding_mask=src_pad_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed Forward
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt