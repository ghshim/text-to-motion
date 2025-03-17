import torch
import torch.nn as nn
import clip

from models.net.encoder import Encoder
from models.net.decoder import Decoder

class MotionTransformer(nn.Module):
    def __init__(
            self,
            num_heads=8, 
            dim_model=512, 
            num_tokens=263,          
            dim_text=512,
            dim_motion=263,         
            num_encoder_layers=8, 
            num_decoder_layers=8, 
            dropout_p=0.1,
            clip_version=None,
            device=None
        ):
        super().__init__()
        self.device = device
        self.clip_model = self.load_and_freeze_clip(clip_version)

        self.text_embedding = nn.Linear(in_features=dim_text, out_features=dim_model)
        self.motion_embedding = nn.Linear(in_features=dim_motion, out_features=dim_model)

        self.encoder = Encoder(
            dim_model=dim_model,
            num_heads=num_heads,
            dropout_p=dropout_p,
            num_layers=num_encoder_layers,
        )
        
        self.decoder = Decoder(
            dim_model=dim_model,
            num_heads=num_heads,
            dropout_p=dropout_p,
            num_layers=num_decoder_layers,
        )
        
        self.out = nn.Linear(dim_model, num_tokens)
        
    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
    
    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device=self.device,
                                                jit=False)  # Must set jit=False for training
        # Added support for cpu
        if str(self.device) != "cpu":
            clip.model.convert_weights(
                clip_model)  # Actually this line is unnecessary since clip by default already on float16
            # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def forward(self, text, tgt, tgt_mask=None, tgt_pad_mask=None, src_pad_mask=None):
        '''
        text
        tgt
        '''
        with torch.no_grad():
            cond_vector = self.encode_text(text)
         
        if tgt_pad_mask is None:
            tgt_pad_mask = self.get_tgt_pad_mask(tgt)
        
        if tgt_mask is None:
            tgt_mask = self.get_tgt_mask(tgt)
        
        # Embedding for Transformer encoder
        src = self.text_embedding(cond_vector)
        
        '''Encoder'''
        memory = self.encoder(src, src_pad_mask=src_pad_mask)
        
        '''Decoder'''
        tgt = self.motion_embedding(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask) # (seq_len, bs, dim_model)
        output = self.out(output)           # (seq_len, bs, num_tokens)
        output = output.permute(1, 0, 2)

        return output
    
    def generate(self, text, src_pad_mask=None, eom_thres=0.025, seq_len=None, Q=5):
        # Make clip text embedding
        with torch.no_grad():
            cond_vector = self.encode_text(text)

        # Embedding for Transformer encoder
        src = self.text_embedding(cond_vector)
        
        '''Encoder'''
        memory = self.encoder(src, src_pad_mask=src_pad_mask)

        bs = len(text)
        predictions = torch.empty((bs, 0, 263), device=self.device)

        SoM = torch.zeros((bs, 1, 263), device=self.device) # Start of Motion
        curr_pose = SoM

        if seq_len is not None:
            if not isinstance(seq_len, torch.Tensor):
                seq_len = torch.tensor(seq_len, device=self.device)

        # predict next possible motion
        i = 0
        while True:
            # if seq_len is not None and i == seq_len: # for training and validation
            #     break
            # For training and validation mode: Stop at seq_len per batch item
            if seq_len is not None and torch.all(i >= seq_len):
                break
            
            if seq_len is None and i >= Q:   # for inference
                q_frames = predictions[:,-Q:,:]
                mean_frames = q_frames.mean(dim=1)
                variance = 1/Q * torch.sum(torch.norm(q_frames - mean_frames, p=2, dim=2), dim=1) # L2 norm 
                # if variance < eom_thres:
                #     # EoM is detected.
                #     print(f"EoM is detected. sequence length: {i}")
                #     break
                # if i > 200:
                #     # EoM is not detected.
                #     print(f"EoM is not detected. Force generating motion.")
                #     break

                if torch.all(variance < eom_thres):
                    print(f"EoM detected. Sequence length: {i}")
                    break
                
                if i > 200:
                    print(f"EoM not detected. Force generating motion.")
                    break
                
            tgt = self.motion_embedding(curr_pose)
            output = self.decoder(tgt, memory)
            output = self.out(output.detach())
            output = output.permute(1, 0, 2)

            # Concatenate current pose to predictions tensor along the sequence length dimension
            curr_pose = output
            predictions = torch.cat([predictions, curr_pose], dim=1) 
            
            i += 1

        return predictions
    
    
    def get_tgt_pad_mask(self, motion):
        '''
        motion: (bs, max_motion_length, num_features)
        tgt_pad_mask: (bs, max_motion_length)
        '''
        tgt_pad_mask = ((motion == 0) | (motion == -1)).all(dim=2) # (bs, max_motion_length, num_features)
        tgt_pad_mask = tgt_pad_mask.float()
        tgt_pad_mask = tgt_pad_mask.masked_fill(tgt_pad_mask == 0, float(0.0))
        tgt_pad_mask = tgt_pad_mask.masked_fill(tgt_pad_mask == 1, float('-inf')) 
        
        return tgt_pad_mask

    def get_tgt_mask(self, tgt):
        '''
        Make (max_motion_length, max_motion_length) target mask
        '''
        size = tgt.size(1)                               # max_motion_length
        mask = torch.tril(torch.ones(size, size) == 1)   # (max_motion_length, max_motion_length)
        mask = mask.float()
        mask = mask.masked_fill(mask==0, float('-inf'))  # convert zeros to -inf
        mask = mask.masked_fill(mask==1, float(0.0))     # convert ones to 0

        return mask
    