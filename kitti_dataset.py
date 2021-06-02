# adapted from https://github.com/nianticlabs/monodepth2/blob/master/datasets/mono_dataset.py

import torch
from torchvision import transforms
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset
from torchvision.io import read_image
import random

from PIL import Image
import numpy as np

class JointRandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image_pair):
        image_l, image_r = image_pair
        
        # ensure horizontal flips are geometrically consistent
        if random.random() > self.p:
            return (tvF.hflip(image_r), tvF.hflip(image_l))
        else:
            return (image_l, image_r)
        
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
        color_jitter = JointColorAugmentation(brightness, contrast, saturation, hue)
        horizontal_flip = JointRandomFlip()
        self.transform = transforms.Compose([color_jitter, horizontal_flip])
        
        self.K = np.array([[718.856 ,   0.    , 607.1928],
                           [  0.    , 718.856 , 185.2157],
                           [  0.    ,   0.    ,   1.    ]])
        self.b = 0.5372
        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        split_names = self.filenames[index].split()
        
        image_l = self.resizer(read_image(self.dataset_path + split_names[0]))
        image_r = self.resizer(read_image(self.dataset_path + split_names[1]))
        
        return self.transform((image_l, image_r))        



