import torch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_patch_subsequent_mask(T, N):
    attn_mask = (torch.triu(torch.ones(T, T)) == 1).transpose(0, 1)
    attn_mask = attn_mask.repeat_interleave(N).view(T, -1)
    attn_mask = attn_mask.T.repeat_interleave(N).view(T * N, -1).T
    return attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))


def generate_prefix_mask(sz, prefix_size):
    mask = torch.zeros(sz, sz)
    mask[:, :prefix_size] = 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_conditional_mask(mask, first_patch_idx, num_heads):

    B, T, N = mask.shape
    attn_mask = mask.view(B, -1)
    B, num_patches = attn_mask.shape
    if first_patch_idx:
        num_patches += 1
        attn_mask = torch.cat([torch.zeros(B, 1), attn_mask], axis=1)

    #attn_mask = attn_mask[..., None].repeat(1, 1, num_patches) * attn_mask[:, None]
    attn_mask = attn_mask[..., None].repeat(1, 1, num_patches).transpose(1,2)
    attn_mask = attn_mask.repeat(num_heads, 1, 1).to(mask.device)
    attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
    return  attn_mask


def generate_learnable_mask(mask, first_patch_idx, num_heads, attn_token):
    """
    CAUSAL + Linear(Attn_Mask)

    :param mask:
    :param first_patch_idx:
    :param num_heads:
    :return:
    """
    B, T, N = mask.shape
    attn_mask = mask.view(B, -1)
    B, num_patches = attn_mask.shape
    if first_patch_idx:
        num_patches += 1
        attn_mask = torch.cat([torch.zeros(B, 1), attn_mask], axis=1)

    attn_token = attn_token(attn_mask.float())  # Learn the contribution over the learned token
    st_token = attn_token[..., None].repeat(1, 1, num_patches) * attn_token[:, None]

    st_mask = (torch.triu(torch.ones(T, T)) == 1).transpose(0, 1)
    st_mask = st_mask.repeat_interleave(N).view(T, -1)
    st_mask = st_mask.T.repeat_interleave(N).view(T * N, -1).T
    st_mask = st_mask.float().masked_fill(st_mask == 0, float('-inf')).masked_fill(st_mask == 1, float(0.0))
    st_mask = st_mask.repeat(B, 1, 1).to(mask.device)
    attn_mask = st_mask + st_token
    return attn_mask.repeat(num_heads, 1, 1)


def generate_attention_mask(mask, mask_type, sz, first_patch_idx=False, num_heads=4, attn_token=None, prefix_size=10, do_reshape=False):
    """

    :param mask: [B, T, len(limbs)] being len(limbs) the number of patches per pose
    num_patches = T*P + (cls_token)
    :return: [B, num_patches, num_patches] -- atention mask with value 0 or -inf
    """
    B, T, N = mask.shape
    if mask_type == "causal":
        attn_mask = generate_square_subsequent_mask(sz)

    elif mask_type == "causal_patch":
        attn_mask = generate_patch_subsequent_mask(T, N)

    elif mask_type == "conditional":
        attn_mask = generate_conditional_mask(mask, first_patch_idx, num_heads)
        if do_reshape:
            attn_mask = attn_mask.reshape(B, num_heads, T, T)

    elif mask_type == "learnable":
        attn_mask = generate_learnable_mask(mask, first_patch_idx, num_heads, attn_token)

    elif mask_type == "prefix":
        attn_mask = generate_prefix_mask(T, prefix_size)
    else:
        return None
    return attn_mask.to(mask.device)


if __name__ == '__main__':

    from utils.load_and_save import read_and_arrange_cfg
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from dataset.H36M_dataset.h36dataset import H36Dataset

    cfg = read_and_arrange_cfg(config_dir="../../config", modelname="MTT", dataset_name="H36M", data_repr="euler")
    cfg_train = cfg["TRAIN"]
    cfg_train["BATCH_SIZE"] = 8
    cfg_train["MASK"] = "leg"  # ["random", "causal", "body", "leg", "arm"]


    print('[DATASET]... Loading Training set from H36M')
    dataset = H36Dataset(
        subject_ids=None,
        split="test",
        cfg=cfg_train)

    loader = DataLoader(dataset,
                              batch_size=cfg_train["BATCH_SIZE"],
                              num_workers=cfg_train["WORKERS"],
                              shuffle=False,
                              pin_memory=False,
                              drop_last=False)

    batch = next(iter(loader))
    data, targets, act_id, patch_mask, src_mask = batch

    print(data.shape)
    B, T, P = data.shape

    attn_mask_type = ["none", "causal", "causal_patch", "conditional", "learnable", "prefix"]


    attn_token = None

    for attn_mask_t in attn_mask_type:

        if attn_mask_t =="learnable":
            B, T, P = src_mask.shape
            total_num_patches =T*P
            attn_token = torch.nn.Linear(total_num_patches, total_num_patches)

        attn_mask = generate_attention_mask(
            mask=src_mask, mask_type=attn_mask_t, sz=T, first_patch_idx=False, num_heads=4, attn_token=attn_token, prefix_size=10
        )
        if attn_mask_t =="learnable":
            attn_mask = attn_mask.detach().numpy()
        if attn_mask is not None:
            print("Attention mask of type {}:".format(attn_mask_t), attn_mask.shape)
            if len(attn_mask.shape)>2:
                attn_mask = attn_mask[0]
            plt.matshow(attn_mask)
            plt.show()
    print("Finished")