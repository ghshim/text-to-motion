import argparse
import os
import torch

class AugmentOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_root', type=str, default='./dataset/HumanML3D', help='Dataset root directory')
        
        self.initialized = True

    def parse(self, extra_options=None):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        if extra_options:
            for key, value in extra_options.items():
                setattr(self.opt, key, value)

        if self.opt.gpu_id != -1:
            torch.cuda.set_device(self.opt.gpu_id)

        # print all options
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt