import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry
import kornia.losses

import numpy as np

def disparity_to_depth(disparity, min_depth=0.1, max_depth=100, stereo_scale_factor=1.0):
    min_disp = 1/max_depth
    max_disp = 1/min_depth
    scaled_disparity = min_disp + (max_disp - min_disp) * disparity
    depth = stereo_scale_factor * 1/scaled_disparity
    
    return depth

class StereoDepthLoss(object):
    def __init__(self, K, b, height, width, batch_size, alpha=0.85, lambda_ = 0.01):
        self.T_rl = np.identity(4)
        self.T_rl[0, 3] = -b
        self.T_lr = -self.T_rl.copy()
        self.T_lr = torch.from_numpy(self.T_lr).repeat(batch_size, 1, 1)
        self.T_rl = torch.from_numpy(self.T_rl).repeat(batch_size, 1, 1)

        self.K = torch.from_numpy(K).repeat(batch_size, 1, 1)
        self.height = height
        self.width = width
        self.alpha = alpha
        self.lambda_ = lambda_
        
    def get_predicted_right(self, image_l, disparity_l):
        depth_l = disparity_to_depth(disparity_l)
        image_r_predicted = kornia.geometry.warp_frame_depth(image_l, depth_l, self.T_rl, self.K)
        
        return image_r_predicted
        
    def get_predicted_left(self, image_r, disparity_r):
        depth_r = disparity_to_depth(disparity_r)
        image_l_predicted = kornia.geometry.warp_frame_depth(image_r, depth_r, self.T_lr, self.K)
        
        return image_l_predicted

    def generate_predicted_images(self, inputs, outputs):
        image_l, image_r = inputs
        disparity_l, disparity_r = outputs
        
        image_r_predicted = self.get_predicted_right(image_l, disparity_l)
        image_l_predicted = self.get_predicted_left(image_r, disparity_r)

        return (image_l_predicted, image_r_predicted)

    def calculate_loss(self, inputs, outputs, predicted):
        image_l, image_r = inputs
        image_l_predicted, image_r_predicted = predicted
        disparity_l, disparity_r = outputs
                
        # Left Photometric Consistency
        reprojection_loss = self.alpha*kornia.losses.ssim_loss(image_l, image_l_predicted, 3)
        reprojection_loss += (1 - self.alpha)*torch.mean(torch.abs(image_l - image_l_predicted))
        
        # Right Photometric Consistency
        reprojection_loss += kornia.losses.ssim_loss(image_r, image_r_predicted, 3)
        reprojection_loss += (1 - self.alpha)*torch.mean(torch.abs(image_r - image_r_predicted))
        
        # Inverse Depth Smoothness Loss
        smoothness_loss = kornia.losses.inverse_depth_smoothness_loss(disparity_l, image_l.float())
        smoothness_loss += kornia.losses.inverse_depth_smoothness_loss(disparity_r, image_r.float())

        losses = {}
        losses["smoothness"] = smoothness_loss
        losses["reprojection"] = reprojection_loss
        losses["total_loss"] = reprojection_loss + self.lambda_*smoothness_loss
        
        return losses
        