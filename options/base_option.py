import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, required=True, default=f'None', help='Name of this trial')
        
        self.parser.add_argument('--data_dir', type=str, default='./dataset/HumanML3D', help='Directory for training data')
        self.parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name, {t2m} for humanml3d')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here.')
        self.parser.add_argument("--max_motion_length", type=int, default=199, help="Max length of motion")
        self.parser.add_argument("--unit_length", type=int, default=4, help="Downscale ratio of VQ")
 
        # self.parser.add_argument("--exp_num", type=str, required=True, help="Experiment number")
        # self.parser.add_argument("--config_path", type=str, required=True, help="Config path")
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        self.opt.is_train = self.is_train
        
        self.opt.save_root = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
        if self.opt.is_train and self.opt.resume_path is None:
            if os.path.exists(self.opt.save_root):
                suffix = 1
                while True:
                    new_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, f"{self.opt.name}-{suffix}")
                    if not os.path.exists(new_path):
                        self.opt.name = f"{self.opt.name}-{suffix}"
                        self.opt.save_root = new_path
                        break
                    suffix += 1

        self.opt.model_dir = os.path.join(self.opt.save_root, 'model')
        self.opt.meta_dir = os.path.join(self.opt.save_root, 'meta')
        self.opt.eval_dir = os.path.join(self.opt.save_root, 'animation')
        self.opt.log_dir = os.path.join(self.opt.save_root, 'log')
        
        os.makedirs(self.opt.model_dir, exist_ok=True)
        os.makedirs(self.opt.meta_dir, exist_ok=True)
        os.makedirs(self.opt.eval_dir, exist_ok=True)
        os.makedirs(self.opt.log_dir, exist_ok=True)

        if self.opt.is_train:
            self.opt.debug_path = os.path.join(self.opt.save_root, 'debug')
            if self.opt.resume_path is None:
                os.makedirs(self.opt.debug_path)
        
        if self.opt.gpu_id != -1:
            # self.opt.gpu_id = int(self.opt.gpu_id)
            torch.cuda.set_device(self.opt.gpu_id)

        # if self.opt.config_path:
        #     self.update_with_config(self.opt.config_path)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if self.is_train:
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
    
    def update_with_config(self, config_path):
        # Flatten the nested config dictionary and update options
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        """ Update options with values from a config file (YAML) """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        flat_config = flatten_dict(config)
    
        for key, value in flat_config.items():
            if hasattr(self.opt, key):
                setattr(self.opt, key, value)
            else:
                print(f"Warning: Option '{key}' not recognized. Skipping.")