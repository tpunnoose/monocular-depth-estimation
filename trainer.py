# adapted from https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

from loss import StereoDepthLoss, disparity_to_depth
from resnet_encoder import ResnetEncoder
from depth_decoder import DepthDecoder

from kitti_dataset import KittiStereoDataset

import os
import time
import json

from utils import *

class Trainer:
    def __init__(self, options):
        self.opt = options

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.encoder = ResnetEncoder(self.opt.num_layers)
        self.encoder.to(self.device)

        self.decoder = DepthDecoder(self.encoder.num_ch_enc, self.opt.scales)
        self.decoder.to(self.device)

        self.training_parameters = []
        self.training_parameters += self.encoder.parameters()
        self.training_parameters += self.decoder.parameters()

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.optimizer = optim.Adam(self.training_parameters, self.opt.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.opt.scheduler_step_size, 0.1)
        
        fpath = os.path.join(os.path.dirname(__file__), "filenames", "{}_".format(self.opt.split)+"{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        
        if self.opt.training_subset != None:
            train_filenames = train_filenames[:self.opt.training_subset]
            
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        val_filenames = readlines(fpath.format("val"))
        
        train_dataset = KittiStereoDataset(self.opt.data_path, train_filenames, is_train=True, height=self.opt.height, width=self.opt.width)
        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers, 
                                       pin_memory=True, drop_last = True)
        
        val_dataset = KittiStereoDataset(self.opt.data_path, val_filenames, is_train=False, height=self.opt.height, width=self.opt.width)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers, 
                                       pin_memory=True, drop_last = True)

        self.val_iter = iter(self.val_loader)
        
        self.stereo_loss = StereoDepthLoss(train_dataset.K, train_dataset.b, self.opt.height, self.opt.width, self.opt.batch_size, self.device, lambda_ = self.opt.disparity_smoothness)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        self.encoder.train()
        self.decoder.train()

    def set_eval(self):
        self.encoder.train()
        self.decoder.train()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    # def val(self):
    #     """Validate the model on a single minibatch
    #     """
    #     self.set_eval()
    #     try:
    #         inputs = self.val_iter.next()
    #     except StopIteration:
    #         self.val_iter = iter(self.val_loader)
    #         inputs = self.val_iter.next()

    #     with torch.no_grad():
    #         outputs, losses = self.process_batch(inputs)

    #         if "depth_gt" in inputs:
    #             self.compute_depth_losses(inputs, outputs, losses)

    #         self.log("val", inputs, outputs, losses)
    #         del inputs, outputs, losses

    #     self.set_train()

    def run_epoch(self):
        if self.epoch > 0:
            self.lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            losses, outputs, predicted = self.process_batch(inputs)

            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["total_loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, predicted, losses)
                # self.val()

            self.step += 1

    def process_batch(self, inputs):
        inputs[0] = inputs[0].to(self.device)
        inputs[1] = inputs[1].to(self.device)
        image_l, image_r = inputs

        features_l = self.encoder(image_l)
        disparities_l = self.decoder(features_l)
        disparity_l_full_size = disparities_l[("disp", 0)]

        features_r = self.encoder(image_r)
        disparities_r = self.decoder(features_r)
        disparity_r_full_size = disparities_r[("disp", 0)]

        outputs = (disparity_l_full_size, disparity_r_full_size)

        predicted = self.stereo_loss.generate_predicted_images(inputs, outputs)
        losses = self.stereo_loss.calculate_loss(inputs, outputs, predicted)

        return losses, outputs, predicted

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        
    def log(self, mode, inputs, outputs, predicted, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        image_l, image_r = inputs
        predicted_l, predicted_r = predicted
        disp_l, disp_r = outputs

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            writer.add_image(
                "left/{}".format(j),
                image_l[j].data, self.step)
            writer.add_image(
                "right/{}".format(j),
                image_r[j].data, self.step)
            writer.add_image(
                "left_pred/{}".format(j),
                normalize_image(predicted_l[j].data), self.step)
            writer.add_image(
                "right_pred/{}".format(j),
                normalize_image(predicted_r[j].data), self.step)
            writer.add_image(
                "disp_left/{}".format(j),
                normalize_image(disp_l[j]), self.step)
            writer.add_image(
                "disp_right/{}".format(j),
                normalize_image(disp_r[j]), self.step)
    
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name in ["encoder", "decoder"]:
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            if model_name == 'encoder':
                to_save = self.encoder.state_dict()
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            else:
                to_save = self.decoder.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for model_name in ["encoder", "decoder"]:
            print("Loading {} weights...".format(model_name))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(model_name))
            model_dict = self.encoder.state_dict() if model_name == "encoder" else self.decoder.state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            if model_name == "encoder":
                self.encoder.load_state_dict(model_dict)
            else:
                self.decoder.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
