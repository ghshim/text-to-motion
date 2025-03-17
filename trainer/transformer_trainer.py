import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.train_utils import wandb_init
from models.transformer import MotionTransformer
from models.module.scheduler.CosineAnnealingWarmRestarts import CosineAnnealingWarmUpRestarts
from evaluator.eval_t2m import evaluate_transformer

class T2MTransformerTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = self.opt.device
        self.model = self.load_model()

    def load_model(self):
        model = MotionTransformer(
            num_heads=self.opt.num_heads,
            dim_model=self.opt.dim_model,
            num_tokens=self.opt.num_tokens,
            dim_text=self.opt.dim_text,
            dim_motion=self.opt.dim_motion,
            num_encoder_layers=self.opt.num_encoder_layers,
            num_decoder_layers=self.opt.num_decoder_layers,
            dropout_p=self.opt.dropout_p,
            clip_version='ViT-B/32',
            device=self.opt.device
        ).to(self.device)

        return model
    
    def train(self, train_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        debug_path_ = os.path.join(self.opt.debug_path, 'train')
        os.makedirs(debug_path_, exist_ok=True)

        if self.opt.resume_path and os.path.exists(self.opt.resume_path):
            wandb_init("Text2Motion", config=self.opt, id=self.opt.name, resume='allow') # TODO: id setting
        else:
            wandb_init("Text2Motion", config=self.opt, id=self.opt.name)

        ## for training
        start_epoch = 0
        num_epochs = self.opt.epochs
        lr = float(self.opt.lr)
        mse = nn.MSELoss(reduction='none')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.0001, T_up=10, gamma=0.5, last_epoch=-1)

        ## for evaluation
        best_fid = 100; best_div = 100; best_top1 = 0; best_top2 = 0; best_top3 = 0; best_matching = 100

        #--------------------------------------------------#
        # Training Start !!
        #--------------------------------------------------#
        if self.opt.resume_path and os.path.exists(self.opt.resume_path):
            checkpoint = torch.load(self.opt.resume_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] 

        initial_eps = 1.0
        lambda_decay = 0.1
        for epoch in tqdm(range(start_epoch+1, num_epochs+1), desc='Train'):
            ''''Training'''
            eps = initial_eps * torch.exp(torch.tensor(-lambda_decay * epoch)).item()
            sample = torch.rand(1).item()   # sample random number between 0 and 1
            
            total_loss = 0.0
            
            for batch_data in train_loader:
                caption, motion, m_length = batch_data

                bs, max_seq_len = motion.shape[0], motion.shape[1]
                motion, m_length = (
                    motion.float().to(self.device),
                    m_length.int().to(self.device)
                )

                #------------- Teacher Forcing ------------#
                teacher_output = self.model(caption, motion[:,:-1,:])
                
                ## Calculate loss
                keep_mask = torch.zeros_like(motion[:,1:,0], dtype=torch.bool) # (bs, max_motion_length-1)
                
                # Make padding mask
                for i, length in enumerate(m_length):
                    keep_mask[i, :length] = True
                masked_gt = motion[:,1:,:] * keep_mask.unsqueeze(-1) # (bs, seq_len-1, num_features)
                
                # Masked loss for teacher forcing output
                pred_masked_teacher = teacher_output * keep_mask.unsqueeze(-1)  # (bs, seq_len-1, num_features)
                # Compute MSE loss (reduction='none')
                mse_loss = mse(pred_masked_teacher, masked_gt)  # (bs, seq_len-1, num_features)
                mse_loss = mse_loss * keep_mask.unsqueeze(-1)   # Remove padding influence
                # Normalize by valid sequence length * num_features
                teacher_loss = (mse_loss.sum(dim=(1,2)) / (m_length * motion.shape[-1])).mean()
                
                if eps > sample:
                    loss = teacher_loss

                #------------- Student Learning ------------#
                else:
                    student_output = self.model.generate(caption, seq_len=max_seq_len-1)

                    ## Calculate loss
                    # Masked loss for student learning output
                    pred_masked_studnet = student_output * keep_mask.unsqueeze(-1)
                    # Compute MSE loss (reduction='none)
                    mse_loss = mse(pred_masked_studnet, masked_gt)  # (bs, seq_len-1, num_features)
                    mse_loss = mse_loss * keep_mask.unsqueeze(-1)   # Remove padding influence
                    # Normalize by valid sequence length * num_features
                    student_loss =  (mse_loss.sum(dim=(1,2)) / (m_length * motion.shape[-1])).mean()

                    loss = (teacher_loss + student_loss) / 2


                total_loss += loss.item()            

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            scheduler.step()

            '''Validation (Evaluation)'''
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching = evaluate_transformer(
                self.opt.save_root, eval_val_loader, self.model, epoch, 
                best_fid=best_fid, best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, 
                eval_wrapper=eval_wrapper, plot_func=plot_eval, save_anim=(epoch%self.opt.eval_every_e==0)
            )

            # Save log
            print(f"Train loss: {train_loss:.4f}")
            wandb.log({"Train Loss": train_loss, "Epsilon": eps, "Sample": sample, "Epoch": epoch})

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict()
                }, os.path.join(debug_path_, f'checkpoint_{epoch:03d}.pth'))

        wandb.finish()
        torch.save(self.model.state_dict(), os.path.join(debug_path_, f'last.pt'))
        
        return self.model
            