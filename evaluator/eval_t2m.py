import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm

from utils.metrics import *
from utils.motion_process import recover_from_ric
from utils.viz import plot_3d_motion_global
from utils.paramUtil import t2m_kinematic_chain

@torch.no_grad()
def evaluate_transformer(out_dir, val_loader, model, epoch, 
                            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 
                            eval_wrapper, plot_func=None, save_anim=False):
    model.eval()

    motion_gt_list = []
    motion_pred_list = []
    R_precision_gt = 0
    R_precision_pred = 0
    matching_score_gt = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, captions, sent_len, motions, m_length, token = batch
        bs, seq = motions.shape[:2]

        motions, m_length = (
            motions.float().cuda(),
            m_length.int().cuda()
        )

        pred_motions = model.generate(captions, seq_len=seq-1)
        # pred_motions = model(captions, motions)
        
        ## get co embeddings between motion and text
        # gt
        et_gt, em_gt = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motions.clone(), m_length)
        # pred
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length)
        
        motion_gt_list.append(em_gt) # gt
        motion_pred_list.append(em_pred)  # pred

        ## Calculate R-Precision
        # gt
        temp_R = calculate_R_precision(et_gt.cpu().numpy(), em_gt.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_gt.cpu().numpy(), em_gt.cpu().numpy()).trace()
        R_precision_gt += temp_R
        matching_score_gt += temp_match
        
        # pred
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision_pred += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_gt_np = torch.cat(motion_gt_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
   
    gt_mu, gt_cov = calculate_activation_statistics(motion_gt_np)
    pred_mu, pred_cov = calculate_activation_statistics(motion_pred_np)

    diversity_gt = calculate_diversity(motion_gt_np, 300 if nb_sample > 300 else 100)
    diversity_pred = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    
    R_precision_gt = R_precision_gt / nb_sample
    R_precision_pred = R_precision_pred / nb_sample

    matching_score_gt = matching_score_gt / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)

    msg = f"--> \t Eva. Epoch {epoch} :, FID. {fid:.4f}, Diversity_gt. {diversity_gt:.4f}, Diversity_pred. {diversity_pred:.4f}, R_precision_gt. {np.round(R_precision_gt, 4)}, R_precision_pred. {np.round(R_precision_pred, 4)}, matching_score_gt. {matching_score_gt:.4f}, matching_score_pred. {matching_score_pred:.4f}"
    print(msg)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_epoch = fid, epoch
        torch.save(model.state_dict(), os.path.join(out_dir, 'model', f'best_fid_E{best_epoch:04d}.pt'))
    
    if abs(diversity_gt - diversity_pred) < abs(diversity_gt - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity_pred:.5f} !!!"
        print(msg)
        best_div = diversity_pred

    if R_precision_pred[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision_pred[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision_pred[0]

    if R_precision_pred[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision_pred[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision_pred[1]

    if R_precision_pred[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision_pred[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision_pred[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if save_anim:
        rand_idx = [0, 1, 2, 3, 4] #torch.randint(bs, (5,))
        pred_data = pred_motions[rand_idx].detach().cpu().numpy()
        gt_data = motions[rand_idx].detach().cpu().numpy()
        
        captions = [captions[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', f'E{epoch:04d}')
        os.makedirs(save_dir, exist_ok=True)

        plot_func(pred_data, gt_data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_encoder_test(out_dir, val_loader, encoder, repeat_id, 
                            eval_wrapper, plot_func=None, save_anim=False):
    encoder.eval()
    
    # gt_mean = np.load('./dataset/HumanML3D/Mean.npy')
    # gt_std = np.load('./dataset/HumanML3D/Std.npy')
    
    gt_mean = np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
    gt_std = np.load('./checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')

    new_mean = np.load('./dataset/HumanML3D/new_Mean.npy')
    new_std = np.load('./dataset/HumanML3D/new_Std.npy')

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    for i, batch in tqdm(enumerate(val_loader), desc='Evaluation'):
        word_embeddings, pos_one_hots, sentences, sent_len, gt_motions, new_motions, embeddings, m_length, token = batch
        bs, seq = gt_motions.shape[:2]

        m_length, gt_motions, new_motions, embeddings = (
            m_length.float().cuda(),
            gt_motions.float().cuda(),
            new_motions.float().cuda(),
            embeddings.float().cuda()
        )

        # output of encoder (enhanced motion from new_motions)
        pred_motions = encoder(new_motions, embeddings)
        
        # get co embeddings between motion and text
        # pred
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length)
        # gt
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, gt_motions.clone(), m_length)
        
        # plot_motion_statistics(gt_motions.detach().cpu().numpy())
        # plot_motion_statistics(new_motions.detach().cpu().numpy())

        motion_annotation_list.append(em) # gt
        motion_pred_list.append(em_pred)  # pred

        # gt
        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        
        # pred
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

        # if save_anim and repeat_id == 0:
        #     rand_idx = torch.randint(bs, (3,))
        #     pred_data = pred_motions[rand_idx].detach().cpu().numpy()
        #     gt_data = gt_motions[rand_idx].detach().cpu().numpy()
        #     input_data = new_motions[rand_idx].detach().cpu().numpy()
        #     captions = [sentences[k] for k in rand_idx]
        #     lengths = m_length[rand_idx].cpu().numpy()
        #     save_dir = os.path.join(out_dir, 'animation', f'R{repeat_id:04d}')
        #     os.makedirs(save_dir, exist_ok=True)
        #     plot_func(pred_data, gt_data, input_data, save_dir, captions, lengths)

        if save_anim and i % 10 == 0:
            save_dir = os.path.join(out_dir, 'animation', f'R{repeat_id:04d}')
            os.makedirs(save_dir, exist_ok=True)

            for idx in range(bs):
                pred_data = pred_motions[idx].detach().cpu().numpy()
                gt_data = gt_motions[idx].detach().cpu().numpy()
                input_data = new_motions[idx].detach().cpu().numpy()

                pred_data = pred_data * gt_std + gt_mean
                gt_data = gt_data * gt_std + gt_mean
                input_data = input_data * new_std + new_mean

                caption = sentences[idx]
                length = m_length[idx].cpu().numpy()    

                pred_joint_data = pred_data[:int(length)]
                gt_joint_data = gt_data[:int(length)]
                input_joint_data = input_data[:int(length)]
                
                pred_xyz = recover_from_ric(torch.from_numpy(pred_joint_data).float(), 22).numpy()
                gt_xyz = recover_from_ric(torch.from_numpy(gt_joint_data).float(), 22).numpy()
                input_xyz = recover_from_ric(torch.from_numpy(input_joint_data).float(), 22).numpy()

                pred_save_path = os.path.join(save_dir, '%02d_pred.gif'%(i*bs+idx))
                gt_save_path = os.path.join(save_dir, '%02d_gt.gif'%(i*bs+idx))
                input_save_path = os.path.join(save_dir, '%02d_input.gif'%(i*bs+idx))
                
                # print(joint.shape)
                plot_3d_motion_global(pred_save_path, t2m_kinematic_chain, pred_xyz, title=caption, fps=20, radius=4)
                plot_3d_motion_global(gt_save_path, t2m_kinematic_chain, gt_xyz, title=caption, fps=20, radius=4)
                plot_3d_motion_global(input_save_path, t2m_kinematic_chain, input_xyz, title=caption, fps=20, radius=4)

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
   
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
    print(msg)

    return fid, R_precision, matching_score_pred