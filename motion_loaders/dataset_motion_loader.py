import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader

# from src.data.tm_dataset import TextMotionDatasetEval, collate_fn
from data.dataset import Text2MotionDatasetEval, collate_fn
from utils.word_vectorizer import WordVectorizer

def get_dataset_motion_loader(opt, batch_size, fname):
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 'tm' or opt.dataset_name == 't2m': #or opt.dataset_name == 'kit':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load('./dataset/HumanML3D/Mean.npy')
        std = np.load('./dataset/HumanML3D/Std.npy')
        
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=True,
                                collate_fn=collate_fn, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset