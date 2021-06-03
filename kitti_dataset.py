# adapted from https://github.com/nianticlabs/monodepth2/blob/master/datasets/mono_dataset.py

import torch
from torchvision import transforms
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset
from torchvision.io import read_image
import random

import os

import numpy as np
from kitti_utils import *

import skimage

class JointColorAugmentation(object):
    def __init__(self, brightness, contrast, saturation, hue, p=0.5):
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p
        
    def __call__(self, image_pair):
        image_l, image_r = image_pair
        if random.random() > self.p:
            # apply same color jitter transform to both images!
            params = self.color_jitter.get_params(
                            self.color_jitter.brightness, self.color_jitter.contrast, self.color_jitter.saturation,
                            self.color_jitter.hue)            
            transforms_arr = []
            transforms_arr.append(transforms.Lambda(lambda img: tvF.adjust_brightness(img, params[1])))
            transforms_arr.append(transforms.Lambda(lambda img: tvF.adjust_contrast(img, params[2])))
            transforms_arr.append(transforms.Lambda(lambda img: tvF.adjust_saturation(img, params[3])))
            transforms_arr.append(transforms.Lambda(lambda img: tvF.adjust_hue(img, params[4])))

            transform = transforms.Compose(transforms_arr)
            return (transform(image_l), transform(image_r))
        else:
            return (image_l, image_r)

class KittiStereoDataset(Dataset):
    def __init__(self, dataset_path, filenames, is_train, height = 192, width = 640):
        self.height = height
        self.width = width
        self.filenames = filenames
        self.dataset_path = dataset_path
        self.is_train = is_train

        # Training Transforms
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        
        self.resizer = transforms.Resize((height, width))
        self.color_jitter = JointColorAugmentation(brightness, contrast, saturation, hue)
        
        self.K = np.array([[718.856 ,   0.    , 607.1928],
                           [  0.    , 718.856 , 185.2157],
                           [  0.    ,   0.    ,   1.    ]])
        self.b = 0.5372

        self.full_res_shape = (1242, 375)

        self.load_depth = self.check_depth()
        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        split_names = self.filenames[index].split()
        
        image_l = self.resizer(read_image(self.dataset_path + split_names[0]))
        image_r = self.resizer(read_image(self.dataset_path + split_names[1]))

        do_flip = random.random() > self.p
        color_aug = random.random() > self.p

        if do_flip:
            image_l, image_r = tvF.hflip(image_r), tvF.hflip(image_l)

        inputs["l"] = image_l
        inputs["r"] = image_r

        if color_aug:
            image_l, image_r = self.color_jitter((image_l, image_r))

        inputs["l_color_aug"] = image_l
        inputs["r_color_aug"] = image_r

        if self.load_depth:
            folder = split_names[0]
            depth_gt_a = self.get_depth(folder, 0, 2, do_flip)
            depth_gt_b = self.get_depth(folder, 0, 3, do_flip)
            if do_flip:
                s_a = "r"
                s_b = "l"
            else:
                s_a = "l"
                s_b = "r"
            inputs["depth_gt_" + s_a] = np.expand_dims(depth_gt_a, 0)
            inputs["depth_gt_" + s_b] = np.expand_dims(depth_gt_b, 0)
            
            inputs["depth_gt_l"] = torch.from_numpy(inputs["depth_gt_l"].astype(np.float32))
            inputs["depth_gt_r"] = torch.from_numpy(inputs["depth_gt_r"].astype(np.float32))

        return inputs

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)
    
    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.dataset_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.dataset_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
