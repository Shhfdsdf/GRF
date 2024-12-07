#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from PIL import Image
from utils.general_utils import PILtoTorch
import os, cv2
import torch.nn.functional as F

def process_image(image_path, resolution, ncc_scale):
    image = Image.open(image_path)
    if len(image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image.split()[3], resolution)
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in image.split()[:3]], dim=0)
    else:
        resized_image_rgb = PILtoTorch(image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = PILtoTorch(image, ncc_resolution)
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    return gt_image, gray_image, loaded_mask


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy,image_path, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 ncc_scale=1.0,
                 preload_img=True,
                 data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.resolution = (self.image_width, self.image_height)
        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
            
        self.original_image, self.image_gray, self.mask = None, None, None
        self.preload_img = preload_img
        self.ncc_scale = ncc_scale
        
        if self.preload_img:
            gt_image, gray_image, loaded_mask = process_image(self.image_path, self.resolution, ncc_scale)
            self.original_image = gt_image.to(self.data_device)
            self.original_image_gray = gray_image.to(self.data_device)
            self.mask = loaded_mask        
        
        
        self.zfar = 100.0
        self.znear = 0.01
        self.nearest_id = []
        self.nearest_names = []
        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)        
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height         
        
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T
        
    def get_image(self):
        if self.preload_img:
            return self.original_image_gray.cuda()
        else:
            gt_image, gray_image, _ = process_image(self.image_path, self.resolution, self.ncc_scale)
            return gray_image.cuda()
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

