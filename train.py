# adapted from https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
import torch
import torch.optim as optim
from torch.utils.data import Dataloader
from tensorboardX import SummaryWriter
import numpy as np

from loss import StereoDepthLoss, disparity_to_depth
from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder

import os

class Trainer:
    def __init__(self, options):
        self.opt = options

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.encoder = ResnetEncoder(self.opt.num_layers)
        self.encoder.to(self.device)

        self.decoder = DepthDecoder(self.encoder.num_ch_enc, self.opt.scales)
        self.decoder.to(self.device)

        self.training_parameters = []
        self.training_parameters += self.encoder.parameters()
        self.training_parameters += self.decoder.parameters()

        self.optimizer = optim.Adam(self.training_parameters, self.opt.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.opt.scheduler_step_size, 0.1)
        
        fpath = os.path.join(os.path.dirname(__file__), "filenames", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        
        train_dataset = KittiStereoDataset(data_dir, train_filenames, is_train=True, self.opt.height, self.opt.width)
        self.train_loader = Dataloader(train_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers, 
                                       pin_memory=True, drop_last = True)
        
        val_dataset = KittiStereoDataset(data_dir, val_filenames, is_train=False, self.opt.height, self.opt.width)
        self.train_loader = Dataloader(val_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers, 
                                       pin_memory=True, drop_last = True)
        
        self.stereo_loss = StereoDepthLoss(train_dataset.K, train_dataset.b, self.opt.height, self.opt.width, self.opt.batch_size)


