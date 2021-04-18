# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader_mp import DataLoader
from nets import *
from utils import *


# auto_mask:reduce_min

class SfMLearner(object):
    def __init__(self):
        pass

    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):

            tgt_image, src_image_stack, tgt_image1, src_image_stack1, tgt_image_aug, src_image_stack_aug, \
            tgt_image1_aug, src_image_stack1_aug, intrinsics, essentialmat, match_points_stack, self.flag = loader.load_train_batch()

            match_points1 = match_points_stack[:,:,:4]
            match_points2 = match_points_stack[:,:,4:8]
            match_points3 = match_points_stack[:,:,8:12]
            match_points4 = match_points_stack[:,:,12:16]

            tgt_image = self.preprocess_image(tgt_image)
           
            src_image_stack = self.preprocess_image(src_image_stack)
            tgt_image1 = self.preprocess_image(tgt_image1)
            src_image_stack1 = self.preprocess_image(src_image_stack1)
            tgt_image_aug = self.preprocess_image(tgt_image_aug)
            src_image_stack_aug = self.preprocess_image(src_image_stack_aug)
            tgt_image1_aug = self.preprocess_image(tgt_image1_aug)
            src_image_stack1_aug = self.preprocess_image(src_image_stack1_aug)

        with tf.name_scope("depth_prediction"):

            pred_disp, depth_net_endpoints = disp_net(tgt_image_aug,
                                                      is_training=True)
            pred_depth = [1. / d for d in pred_disp]
            pred_disp1, depth_net_endpoints1 = disp_net(tgt_image1_aug,
                                                        is_training=True,
                                                        isReuse=True)
            pred_depth1 = [1. / d for d in pred_disp1]

        with tf.name_scope("pose"):

            pred_poses, pred_exp_logits, _ = \
                pose_exp_net(tgt_image_aug,
                             src_image_stack_aug,
                             is_training=True)
            pred_poses1, pred_exp_logits, _ = \
                pose_exp_net(tgt_image1_aug,
                             src_image_stack1_aug,
                             is_training=True,
                             isReuse=True)
        with tf.name_scope("compute_loss"):
            pixel_losses = 0
            ssim_loss = 0
            smooth_loss = 0
            pixel_match_loss = 0
            depth_loss = 0
            epipolar_loss = 0
            pre_match_loss = 0
            epipolar_loss_V1 = 0
            epipolar_loss_V2 = 0
            tgt_image_all = []
            src_image_stack_all = []

            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []

            for s in range(opt.num_scales):
                print(s)
                # Scale the source and target images for computing loss at the
                # according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image,
                                                      [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])
                curr_src_image_stack = tf.image.resize_area(src_image_stack,
                                                            [int(opt.img_height / (2 ** s)),
                                                             int(opt.img_width / (2 ** s))])

                curr_tgt_image1 = tf.image.resize_area(tgt_image1,
                                                       [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])
                curr_src_image_stack1 = tf.image.resize_area(src_image_stack1,
                                                             [int(opt.img_height / (2 ** s)),
                                                              int(opt.img_width / (2 ** s))])
                if s == 0:
                    if opt.epipolar_loss_weight > 0:
                        
                        tgt_cloud, _ = get_cloud(
                            tf.squeeze(pred_depth[0][: opt.batch_size, :, :, :], axis=3),
                            intrinsics[:, 0, :, :])
                        tgt_cloud1, _ = get_cloud(
                            tf.squeeze(pred_depth1[0][: opt.batch_size, :, :, :], axis=3),
                            intrinsics[:, 0, :, :])

                        tgt_cloudcloud2, tgt_src_cloud2_1 = transform_cloud_2(tgt_cloud, pred_poses[:, :6],
                                                                                intrinsics[:, 0, :, :])

                        tgt_cloudcloud3, tgt_src_cloud3_4 = transform_cloud_2(tgt_cloud1, pred_poses1[:, 6:],
                                                                                intrinsics[:, 0, :, :], True)

                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight / (2 ** s) * \
                                   self.compute_smooth_loss(pred_disp[s], curr_tgt_image)

                    smooth_loss += opt.smooth_weight / (2 ** s) * \
                                   self.compute_smooth_loss(pred_disp1[s], curr_tgt_image1)

                reprojection_losses = []

                for i in range(opt.num_source):
                    # 2-1 2-3
                    curr_proj_image, curr_proj_depth, mask, computed_depth = projective_inverse_warp_withdepth(
                        curr_src_image_stack[:, :, :, 3 * i:3 * (i + 1)],
                        tf.squeeze(pred_depth[s], axis=3),
                        pred_poses[:, 6 * i:6 * (i + 1)],
                        intrinsics[:, s, :, :],
                        pred_depth1[s])
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    curr_proj_error1 = tf.abs(curr_src_image_stack[:, :, :, 3 * i:3 * (i + 1)] - curr_tgt_image)
                    
                    # mask
                    if opt.cm_mask:
                        promask = tf.cast(
                            tf.reduce_mean(curr_proj_error, axis=3, keep_dims=True) < tf.reduce_mean(curr_proj_error1,
                                                                                                     axis=3,
                                                                                                     keep_dims=True),
                            'float32')
                        mask = tf.clip_by_value(promask, 0, 1.0) * mask
                        pixel_loss = tf.reduce_mean(curr_proj_error * mask)
                        # SSIM loss
                        if opt.ssim_weight > 0:

                            ssim_loss = tf.reduce_mean(
                                mask * self.compute_ssim_loss(curr_proj_image, curr_tgt_image))

                    else:
                        pixel_loss = tf.reduce_mean(curr_proj_error)
                        if opt.ssim_weight > 0:
                            ssim_loss = tf.reduce_mean(self.compute_ssim_loss(curr_proj_image, curr_tgt_image))
                    reprojection_losses = opt.ssim_weight * ssim_loss + (1 - opt.ssim_weight) * pixel_loss
                    pixel_losses += reprojection_losses

                    
                    if s == 0:
                        if opt.match_weight > 0:
                            # Positions of matched feature pairs loss
                            if i == 1:
                                mt = match_points2
                                depth = pred_depth[s]
                                depth1 = pred_depth1[s]
                                tgt_im = curr_tgt_image
                                src_im = curr_src_image_stack[:, :, :, 3:]
                                curr_pose = pred_poses[:, 6:]

                                pre_match_loss += self.compute_matchloss(mt, depth,  depth1, tgt_im, src_im, intrinsics[:, s, :, :], curr_pose)

                        if opt.epipolar_loss_weight > 0:
                            # Epipolar geometry constraint loss
                            if i == 0:
                                tgt_src_cloud = tgt_src_cloud2_1
                                essential_mat = essentialmat[:, 0, :, :]
                                intrinsics_inv = tf.matrix_inverse(intrinsics[:, 0, :, :])
                                fundamental_mat = tf.matmul(intrinsics_inv, essential_mat, transpose_a=True)
                                fundamental_mat = tf.matmul(fundamental_mat, intrinsics_inv)
                                curr_mat = fundamental_mat

                                epipolar_loss_V1 += self.compute_epipolar_error_V1(
                                tgt_cloudcloud2,
                                tgt_src_cloud,
                                curr_mat,
                                mask)  
                            else:
                            # if i == 1:
                                matches = match_points2
                                pose_inverse = get_inverse_R(pred_poses1[:,:6])
                                curr_mat = fundamental_matrix_from_rt(pose_inverse, intrinsics[:, s, :, :])
                                epipolar_loss_V2 += self.compute_epipolar_error_V2(matches, curr_mat, intrinsics[:, s, :, :])

                    if i == 1 and opt.depth_weight > 0:
                        # Point cloud consistency loss
                        depth_loss += tf.reduce_mean(
                                self.compute_depth_loss(curr_proj_depth, computed_depth) * mask)

                for i in range(opt.num_source):
                    # 3-2 3-4
                    curr_proj_image, curr_proj_depth, mask, computed_depth = projective_inverse_warp_withdepth(
                        curr_src_image_stack1[:, :, :, 3 * i:3 * (i + 1)],
                        tf.squeeze(pred_depth1[s], axis=3),
                        pred_poses1[:, 6 * i:6 * (i + 1)],
                        intrinsics[:, s, :, :],
                        pred_depth[s])
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image1)
                    curr_proj_error1 = tf.abs(curr_src_image_stack1[:, :, :, 3 * i:3 * (i + 1)] - curr_tgt_image1)
                    if opt.cm_mask:
                        # mask
                        promask = tf.cast(
                            tf.reduce_mean(curr_proj_error, axis=3, keep_dims=True) < tf.reduce_mean(curr_proj_error1,
                                                                                                     axis=3,
                                                                                                     keep_dims=True),
                            'float32')
                        mask = tf.clip_by_value(promask, 0, 1.0) * mask
                        pixel_loss = tf.reduce_mean(curr_proj_error * mask)
                        # SSIM loss
                        if opt.ssim_weight > 0:
                            ssim_loss = tf.reduce_mean(
                                mask * self.compute_ssim_loss(curr_proj_image, curr_tgt_image1))
                    else:
                        pixel_loss = tf.reduce_mean(curr_proj_error)
                        if opt.ssim_weight > 0:
                            ssim_loss = tf.reduce_mean(self.compute_ssim_loss(curr_proj_image, curr_tgt_image1))
                    reprojection_losses = opt.ssim_weight * ssim_loss + (1 - opt.ssim_weight) * pixel_loss
                    pixel_losses += reprojection_losses
                    if s == 0:
                        if opt.match_weight > 0:
                            if i == 0 :
                                mt = tf.concat([match_points2[:,:,2:4], match_points2[:,:,0:2]], axis=2)
                                depth = pred_depth1[s]
                                depth1 = pred_depth[s]
                                tgt_im = curr_tgt_image1
                                src_im = curr_src_image_stack1[:, :, :, :3]
                                curr_pose = pred_poses1[:, :6]
                                
                            pre_match_loss += self.compute_matchloss(mt, depth,  depth1, tgt_im, src_im, intrinsics[:, s, :, :], curr_pose)

                        if opt.epipolar_loss_weight > 0:
                            if i == 0:
                                matches = tf.concat([match_points2[:,:,2:4], match_points2[:,:,0:2]], axis=2)
                                pose_inverse = get_inverse_R(pred_poses[:,6:])
                                curr_mat = fundamental_matrix_from_rt(pose_inverse, intrinsics[:, s, :, :])

                                epipolar_loss_V2 += self.compute_epipolar_error_V2(matches, curr_mat, intrinsics[:, s, :, :])
                            
                            else:

                                tgt_src_cloud = tgt_src_cloud3_4
                                essential_mat = essentialmat[:, 3, :, :]
                                intrinsics_inv = tf.matrix_inverse(intrinsics[:, 0, :, :])
                                fundamental_mat = tf.matmul(intrinsics_inv, essential_mat, transpose_a=True)
                                fundamental_mat = tf.matmul(fundamental_mat, intrinsics_inv)
                                curr_mat = fundamental_mat

                                epipolar_loss_V1 += self.compute_epipolar_error_V1(
                                    tgt_cloudcloud3,
                                    tgt_src_cloud,
                                    curr_mat,
                                    mask)
                                

                        if i == 0 and opt.depth_weight > 0:
                            depth_loss += tf.reduce_mean(
                                    self.compute_depth_loss(curr_proj_depth, computed_depth) * mask)

                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.cm_mask:
                            exp_mask_stack = mask
                    else:
                        proj_image_stack = tf.concat([proj_image_stack,
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack,
                                                      curr_proj_error], axis=3)
                        if opt.cm_mask:
                            exp_mask_stack = tf.concat([exp_mask_stack,
                                                        mask], axis=3)


                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.cm_mask:
                    exp_mask_stack_all.append(exp_mask_stack)

            train_vars = [var for var in tf.trainable_variables()]
            self.total_step = opt.total_epoch * loader.steps_per_epoch
            self.global_step = tf.Variable(0, name='global_step', trainable=False)  # int32


            #lr
            learning_rates = [opt.start_learning_rate, opt.start_learning_rate / 10]
            boundaries = [np.int(self.total_step * 4 / 5)]
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, learning_rates)


            optimizer = tf.train.AdamOptimizer(self.learning_rate, opt.beta1)
            self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

            depth_loss = depth_loss * opt.depth_weight
            epipolar_loss = (epipolar_loss_V1 + epipolar_loss_V2) * 0.01
            pixel_match_loss = pixel_match_loss   
            pixel_losses /= opt.num_scales

            total_loss = pixel_losses + \
                         smooth_loss + depth_loss + epipolar_loss * opt.epipolar_loss_weight  + pixel_match_loss * opt.match_weight

            self.train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth
        self.steps_per_epoch = loader.steps_per_epoch
        self.pred_poses = pred_poses
        self.pred_poses1 = pred_poses1

        self.total_loss = total_loss
        self.pixel_losses = pixel_losses
        self.ssim_loss = ssim_loss
        self.smooth_loss = smooth_loss
        self.depth_loss = depth_loss
        self.epipolar_loss = epipolar_loss
        self.pixel_match_loss = pixel_match_loss

        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        if opt.cm_mask:
            self.exp_mask_stack_all = exp_mask_stack_all

    def compute_depth_loss(self, proj_depth, computed_depth):
        depth_loss = tf.abs(proj_depth - computed_depth) / (proj_depth + computed_depth)
        return depth_loss

    def compute_ssim_loss(self, x, y):
        """Computes a differentiable structured image similarity measure."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    

    def compute_matchloss(self, pt, depth, depth1, tgt_img, src_img, intrinsics, pose):
        batch = depth.shape[0]
        num = pt.shape[1]
        height = depth.shape[1]
        width = depth.shape[2]
        points1 = tf.slice(pt, [0, 0, 0], [-1, -1, 2]) #pt
        points2 = tf.slice(pt, [0, 0, 2], [-1, -1, 2]) #ps
        
        pixelcloud_zero = tf.tile(tf.zeros(depth[:,:,:-1,:].shape),[1,1,1,2])
        cloud1 = tf.zeros([batch,height - num,2])
        mask_cloud1 = tf.expand_dims(tf.concat([tf.ones(points1.shape), cloud1], axis=1), axis=2)
        mask = tf.concat([pixelcloud_zero, mask_cloud1], axis=2)
        mask = tf.slice(mask, [0, 0, 0, 0], [-1,-1,-1,1])

        cloudtgt_concat = tf.expand_dims(tf.concat([points1, cloud1], axis=1), axis=2)
        pixelcloud_tgt = tf.concat([pixelcloud_zero, cloudtgt_concat], axis=2)
        output_depth_tgt, _ = bilinear_sampler(depth, pixelcloud_tgt, depth)

        cloudsrc_concat = tf.expand_dims(tf.concat([points2, cloud1], axis=1), axis=2)
        pixelcloud_src = tf.concat([pixelcloud_zero, cloudsrc_concat], axis=2)
        

        pixelcloud_tgt = tf.concat([pixelcloud_tgt, tf.ones(depth.shape)], axis=3)
        pixelcloud_tgt = tf.transpose(pixelcloud_tgt, perm=[0, 3, 1, 2])
        

        output_depth_tgt = tf.squeeze(output_depth_tgt)

        cam_coords = pixel2cam(output_depth_tgt, pixelcloud_tgt, intrinsics)  
        pose = pose_vec2mat(pose)
        # # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch, 1, 1])
        intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
        intrinsics = tf.concat([intrinsics, filler], axis=1)
        # # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
        # # pixel frame.
        proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
        src_pixel_coords, computed_depth = cam2pixel_depth(cam_coords, proj_tgt_cam_to_src_pixel)
        # mt_error = tf.abs((src_pixel_coords - pixelcloud_src) / (src_pixel_coords + pixelcloud_src + 1e-6)) * mask

        compute_depth, _ = bilinear_sampler(depth1, src_pixel_coords, depth)
        src_depth, _ = bilinear_sampler(depth1, pixelcloud_src, depth)
        mt_error = tf.abs((computed_depth - src_depth) / (computed_depth + src_depth))

        mt_error *= mask
        mt_error = mt_error[:,0:num,-1,:]
        return tf.reduce_mean(mt_error) / 4


    def compute_epipolar_error_V1(self, p1, p2, fundamental_matrix, mask):
        # r(p1->p2)
        # mask = tf.transpose(mask, [0, 3, 1, 2])
        
        ones = tf.ones(p1.shape)[:,:,:,0:1]
        p1 = tf.concat([p1, ones], axis=3)
        p2 = tf.concat([p2, ones], axis=3)
        
        p1 = tf.transpose(p1, [0, 3, 1, 2])
        mask = tf.transpose(mask, [0, 3, 1, 2])
        
        p2 = tf.transpose(p2, [0, 3, 1, 2])
        p1 = tf.reshape(p1, [self.opt.batch_size, 3, -1])
        mask = tf.reshape(mask, [self.opt.batch_size, 1, -1])

        p1 = tf.expand_dims(tf.transpose(p1, [0, 2, 1]), axis=-1)
        mask = tf.expand_dims(tf.transpose(mask, [0, 2, 1]), axis=-1)

        fundamental_matrix = tf.tile(
            tf.expand_dims(fundamental_matrix, axis=1),
            [1, self.opt.img_height * self.opt.img_width, 1, 1])

        ep1 = tf.matmul(fundamental_matrix, p1[:, :, :, :])

        p2 = tf.reshape(p2, [self.opt.batch_size, 3, -1])
        p2 = tf.expand_dims(tf.transpose(p2, [0, 2, 1]), axis=2)

        epipolar_error = tf.abs(tf.matmul(p2[:, :, :, :], ep1[:, :, :, :]))
        
        return tf.reduce_mean(epipolar_error * mask)

    def compute_epipolar_error_V2(self, matches, curr_mat, intrinsics):
        points1 = tf.slice(matches, [0, 0, 0], [-1, -1, 2]) #pt
        points2 = tf.slice(matches, [0, 0, 2], [-1, -1, 2]) #ps
        num = points1.shape[1]
        ones = tf.ones([self.opt.batch_size, num, 1])
        points1 = tf.concat([points1, ones], axis=2)
        points2 = tf.concat([points2, ones], axis=2)
        match_num = matches.get_shape().as_list()[1]

        fmat = curr_mat

        fmat = tf.expand_dims(fmat, axis=1)
        fmat_tiles = tf.tile(fmat, [1, match_num, 1, 1])
        epi_lines = tf.matmul(fmat_tiles, tf.expand_dims(points1, axis=3))
        dist_p2l = tf.abs(tf.matmul(tf.transpose(epi_lines, perm=[0, 1, 3, 2]), tf.expand_dims(points2, axis=3)))

        a = tf.slice(epi_lines, [0,0,0,0], [-1,-1,1,-1])
        b = tf.slice(epi_lines, [0,0,1,0], [-1,-1,1,-1])
        dist_div = tf.sqrt(a*a + b*b) + 1e-6
        dist_p2l = tf.reduce_mean(dist_p2l / dist_div)
        return dist_p2l

    def compute_smooth_loss(self, disp, img):
        norm_disp = disp / (tf.reduce_mean(disp, [1, 2], keep_dims=True) + 1e-7)

        grad_disp_x = tf.abs(norm_disp[:, :-1, :, :] - norm_disp[:, 1:, :, :])
        grad_disp_y = tf.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

        grad_img_x = tf.abs(img[:, :-1, :, :] - img[:, 1:, :, :])
        grad_img_y = tf.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

        weight_x = tf.exp(-tf.reduce_mean(grad_img_x, 3, keep_dims=True))
        weight_y = tf.exp(-tf.reduce_mean(grad_img_y, 3, keep_dims=True))

        smoothness_x = grad_disp_x * weight_x
        smoothness_y = grad_disp_y * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_losses", self.pixel_losses)
        if opt.depth_weight > 0:
            tf.summary.scalar("depth_loss", self.depth_loss)
        if opt.epipolar_loss_weight > 0:
            tf.summary.scalar("epipolar_loss", self.epipolar_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        if opt.match_weight > 0:
            tf.summary.scalar("pixel_match_loss", self.pixel_match_loss)
        for s in range(opt.num_scales):
            tf.summary.image("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image('scale%d_disparity_image' % s, 1. / self.pred_depth[s])
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
            for i in range(1):
                if opt.cm_mask:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i),
                        tf.expand_dims(self.exp_mask_stack_all[s][:, :, :, i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i),
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i * 3:(i + 1) * 3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i),
                                 self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i * 3:(i + 1) * 3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                                 self.deprocess_image(
                                     tf.clip_by_value(self.proj_error_stack_all[s][:, :, :, i * 3:(i + 1) * 3] - 1, -1,
                                                      1)))

    def train(self, opt):
        self.opt = opt
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        #        opt.num_scales = 4

        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                             for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                    max_to_keep=5)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, self.total_step + 1):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }
                fetches["flag"] = self.flag

                # fetches["incr_x1"] = self.incr_x1

                
                
                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op
                    fetches["lr"] = self.learning_rate

                results = sess.run(fetches)
                gs = results["global_step"]


                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f lr: {%.5f}" \
                          % (train_epoch, train_step, self.steps_per_epoch, \
                             (time.time() - start_time) / opt.summary_freq,
                             results["loss"], results["lr"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                #                if step != 0 and step % (self.steps_per_epoch * 2)  == 0:
                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                                self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(
                input_mc, is_training=False)
            pred_depth = [1. / disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                                self.img_height, self.img_width * self.seq_length, 3],
                                     name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_exp_net(
                tgt_image, src_image_stack, do_exp=False, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. - 1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.) / 2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self,
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs: inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
