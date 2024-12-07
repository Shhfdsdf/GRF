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

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc
from gaussian_renderer import render, network_gui
from utils.graphics_utils import patch_offsets, patch_warp
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import time
import torch.nn.functional as F
from utils.vis_utils import apply_depth_colormap, save_points, colormap
from utils.depth_utils import depths_to_points, depth_to_normal


def culling(xyz, cams, expansion=2):
    cam_centers = torch.stack([c.camera_center for c in cams], 0).to(xyz.device)
    span_x = cam_centers[:, 0].max() - cam_centers[:, 0].min()
    span_y = cam_centers[:, 1].max() - cam_centers[:, 1].min() # smallest span
    span_z = cam_centers[:, 2].max() - cam_centers[:, 2].min()

    scene_center = cam_centers.mean(0)

    span_x = span_x * expansion
    span_y = span_y * expansion
    span_z = span_z * expansion

    x_min = scene_center[0] - span_x / 2
    x_max = scene_center[0] + span_x / 2

    y_min = scene_center[1] - span_y / 2
    y_max = scene_center[1] + span_y / 2

    z_min = scene_center[2] - span_x / 2
    z_max = scene_center[2] + span_x / 2


    valid_mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
                 (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
                 (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
    # print(f'scene mask ratio {valid_mask.sum().item() / valid_mask.shape[0]}')

    return valid_mask, scene_center

def prune_low_contribution_gaussians(gaussians, cameras, pipe, bg, K=5, prune_ratio=0.1,kernel=None):
    top_list = [None, ] * K
    for i, cam in enumerate(cameras):
        trans = render(cam, gaussians, pipe, bg, kernel_size=kernel, record_transmittance=True)
        if top_list[0] is not None:
            m = trans > top_list[0]
            if m.any():
                for i in range(K - 1):
                    top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
                top_list[0][m] = trans[m]
        else:
            top_list = [trans.clone() for _ in range(K)]

    contribution = torch.stack(top_list, dim=-1).mean(-1)
    tile = torch.quantile(contribution, prune_ratio)
    prune_mask = contribution < tile
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()
    
def gen_virtul_cam(cam, trans_noise=1.5, deg_noise=30.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.original_image,
                        cam.gt_alpha_mask,
                        cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        data_device = "cuda")
    return virtul_cam


@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image


def huber_loss(x, delta=1.0):
    return torch.where(torch.abs(x) < delta, 0.5 * x ** 2, delta * (torch.abs(x) - 0.5 * delta))



def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad
    

def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None

    ema_loss_for_log = 0.0
    ema_geo_for_log = 0.0
    ema_ncc_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        gt_image_gray = viewpoint_cam.get_image()    
        
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        image = rendering[:3, :, :]
        
        # rgb Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        # use L1 loss for the transformed image if using decoupled appearance
        if dataset.use_decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)
        
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)
        distortion_loss = distortion_map.mean()
        
        # depth normal consistency
        depth = rendering[6, :, :]
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
        
        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        normal_error = 1 - (render_normal_world * depth_normal).sum(dim=0)
        depth_normal_loss = normal_error.mean()
        
        lambda_distortion = opt.lambda_distortion if iteration >= opt.distortion_from_iter else 0.0
        lambda_depth_normal = opt.lambda_depth_normal if iteration >= opt.depth_normal_from_iter else 0.0

        
        
        # multi-view loss
        geo_loss = torch.tensor(0.0).to(rgb_loss.device)
        ncc_loss = torch.tensor(0.0).to(rgb_loss.device)
        if iteration > opt.multi_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam=False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam=True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                geo_weight = opt.multi_loss_weight
                ncc_weight = opt.multi_view_ncc_weight
                ## compute geometry consistency mask and loss
                H, W = depth.squeeze().shape
                ix, iy = torch.meshgrid( torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(depth.device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)
                rendering_near = nearest_render_pkg["render"]
                pts = depths_to_points(viewpoint_cam,depth)
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, rendering_near[6, :, :], pts_in_nearest_cam)

                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack([pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                                                   pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                # pixel_noise = pixel_noise / (pixel_noise.max() + 1e-6)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()        
                        ## sample mask
                        # 计算光度一致性~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    with torch.no_grad():
                        d_mask = d_mask.reshape(-1)
                        valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                        if d_mask.sum() > sample_num:
                            index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                            valid_indices = valid_indices[index]

                        weights = weights.reshape(-1)[valid_indices]
                        ## sample ref frame patch
                        pixels = pixels.reshape(-1,2)[valid_indices]
                        offsets = patch_offsets(patch_size, pixels.device)
                        
                        ###分母是ncc_scale,部分0.5，360数据集1.0
                        ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                        H, W = gt_image_gray.squeeze().shape
                        pixels_patch = ori_pixels_patch.clone()
                        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                        ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                        ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                        ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                        ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]
                        
                    ## compute Homography
                    render2 = render_pkg["render"]
                    nor = render2[3:6, :, :]
                    
                    ref_local_n = nor.permute(1,2,0)
                    ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]
                    
                    dis = render2[6, :, :]
                    ref_local_d = dis.squeeze()

                    ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                    H_ref_to_neareast = ref_to_neareast_r[None] - \
                        torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                    ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                    H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                    H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                    ## compute neareast frame patch
                    grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                    nearest_image_gray = nearest_cam.get_image()
                    sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                    sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                    ## compute loss
                    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                    mask = ncc_mask.reshape(-1)
                    ncc = ncc.reshape(-1) * weights
                    ncc = ncc[mask].squeeze()

                    if mask.sum() > 0:
                        ncc_loss = ncc_weight * ncc.mean()                        
                        
                        
                        

        loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion + geo_loss + ncc_loss
        loss.backward()
        
        iter_end.record()

        is_save_images = True
        if is_save_images and (iteration % opt.densification_interval == 0):
            with torch.no_grad():
                eval_cam = allCameras[random.randint(0, len(allCameras) -1)]
                
                rendering = render(eval_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size)["render"]
                image = rendering[:3, :, :]
                transformed_image = L1_loss_appearance(image, eval_cam.original_image.cuda(), gaussians, eval_cam.idx, return_transformed_image=True)
                
                normal = rendering[3:6, :, :]
                normal = torch.nn.functional.normalize(normal, p=2, dim=0)
                
            # transform to world space
            c2w = (eval_cam.world_view_transform.T).inverse()
            normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
            normal = normal2.reshape(3, *normal.shape[1:])
            normal = (normal + 1.) / 2.
            
            depth = rendering[6, :, :]
            depth_normal, _ = depth_to_normal(eval_cam, depth[None, ...])
            depth_normal = (depth_normal + 1.) / 2.
            depth_normal = depth_normal.permute(2, 0, 1)
            
            gt_image = eval_cam.original_image.cuda()
            
            depth_map = apply_depth_colormap(depth[..., None], rendering[7, :, :, None], near_plane=None, far_plane=None)
            depth_map = depth_map.permute(2, 0, 1)
            
            accumlated_alpha = rendering[7, :, :, None]
            colored_accum_alpha = apply_depth_colormap(accumlated_alpha, None, near_plane=0.0, far_plane=1.0)
            colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)
            
            distortion_map = rendering[8, :, :]
            distortion_map = colormap(distortion_map.detach().cpu().numpy()).to(normal.device)
        
            row0 = torch.cat([gt_image, image, depth_normal, normal], dim=2)
            row1 = torch.cat([depth_map, colored_accum_alpha, distortion_map, transformed_image], dim=2)
            
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0, 1)
            
            os.makedirs(f"{dataset.model_path}/log_images", exist_ok = True)
            torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/log_images/{iteration}.jpg")
            
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_geo_for_log = 0.4 * geo_loss.item() + 0.6 * ema_geo_for_log
            ema_ncc_for_log = 0.4 * ncc_loss.item() + 0.6 * ema_ncc_for_log
            if iteration % 10 == 0:
                loss_dict ={
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "geo": f"{ema_geo_for_log:.{7}f}",
                    "ncc":f"{ema_ncc_for_log:.{7}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)
                if iteration > opt.contribution_prune_from_iter and iteration % opt.contribution_prune_interval == 0:
                    prune_low_contribution_gaussians(gaussians, allCameras, pipe, background, K=5, prune_ratio=opt.contribution_prune_ratio,kernel=dataset.kernel_size)
                    print(f'Num gs after contribution prune: {len(gaussians.get_xyz)}')
                    gaussians.compute_3D_filter(cameras=trainCameras) 
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    image = rendering[:3, :, :]
                    normal = rendering[3:6, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
