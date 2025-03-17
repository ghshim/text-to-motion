import os
import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader

from options.train_option import TrainT2MOptions
from data.dataset import Text2MotionDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.paramUtil import t2m_kinematic_chain
from utils.fixseed import fixseed
from utils.word_vectorizer import WordVectorizer
from utils.viz import plot_3d_motion_global
from evaluator.t2m_eval_wrapper import EvaluatorModelWrapper

from trainer.transformer_trainer import T2MTransformerTrainer

def plot_t2m(pred_data, gt_data, save_dir, captions, m_lengths):
    pred_data = train_dataset.inv_transform(pred_data)
    gt_data = train_dataset.inv_transform(gt_data)
    
    for i, (caption, pred_joint_data, gt_joint_data) in enumerate(zip(captions, pred_data, gt_data)):
        pred_joint_data = pred_joint_data[:int(m_lengths[i])]
        gt_joint_data = gt_joint_data[:int(m_lengths[i])]
        
        pred_xyz = recover_from_ric(torch.from_numpy(pred_joint_data).float(), 22).numpy()
        gt_xyz = recover_from_ric(torch.from_numpy(gt_joint_data).float(), 22).numpy()
        
        pred_save_path = os.path.join(save_dir, '%02d_pred.gif'%i)
        gt_save_path = os.path.join(save_dir, '%02d_gt.gif'%i)

        plot_3d_motion_global(pred_save_path, t2m_kinematic_chain, pred_xyz, title=caption, fps=fps, radius=radius)
        plot_3d_motion_global(gt_save_path, t2m_kinematic_chain, gt_xyz, title=caption, fps=fps, radius=radius)
        

if __name__ == '__main__':
    parser = TrainT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.data_root = './dataset/HumanML3D'
    opt.motion_dir = os.path.join(opt.data_root, 'new_joint_vecs')
    opt.text_dir = os.path.join(opt.data_root, 'texts')
    opt.joints_num = 22
    opt.max_motion_len = 55
    dim_pose = 263
    radius = 4
    fps = 20
    kinematic_chain = t2m_kinematic_chain
    dataset_opt_path = 'evaluator/Comp_v6_KLD005/opt.txt'
    clip_version = 'ViT-B/32'
    
    '''Load Dataset'''
    train_split_file = os.path.join(opt.data_root, 'train.txt')
    val_split_file = os.path.join(opt.data_root, 'val.txt')
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    mean = np.load('./dataset/HumanML3D/Mean.npy')
    std = np.load('./dataset/HumanML3D/Std.npy')
    
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    # val_dataset = Text2MotionDataset(opt, mean, std, train_split_file, w_vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    
    wrapper_opt = get_opt(dataset_opt_path, device=opt.device)
    eval_val_loader, _ = get_dataset_motion_loader(wrapper_opt, batch_size=64, fname='val')
    print("Loading train/val dataloader Completed!")

    '''Load evaluator'''
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    '''Training'''
    transformer_trainer = T2MTransformerTrainer(opt)
    model = transformer_trainer.train( train_loader, eval_val_loader, eval_wrapper, plot_eval=plot_t2m)
    