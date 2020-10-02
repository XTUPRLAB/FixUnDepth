# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
from utils import *


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
            tgt_image1_aug, src_image_stack1_aug, intrinsics, match, matchmask, essentialmat = loader.load_train_batch()
            # tgt_image, src_image_stack, tgt_image1, src_image_stack1, tgt_image_aug, src_image_stack_aug, \
            # tgt_image1_aug, src_image_stack1_aug, intrinsics, essentialmat = loader.load_train_batch()
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

        with tf.name_scope("pose_and_explainability_prediction"):

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
                        if opt.epipolar_loss_weight > 0:
                            tgt_cloud = get_cloud(
                                tf.squeeze(pred_depth[0][: opt.batch_size, :, :, :], axis=3),
                                intrinsics[:, 0, :, :])
                            tgt_cloud1 = get_cloud(
                                tf.squeeze(pred_depth1[0][: opt.batch_size, :, :, :], axis=3),
                                intrinsics[:, 0, :, :])

                            tgt_src_cloud2_1 = transform_cloud(tgt_cloud, pred_poses[:, :6],False)

                            tgt_src_cloud2_3 = transform_cloud(tgt_cloud, pred_poses1[:, :6])

                            tgt_src_cloud3_2 = transform_cloud(tgt_cloud1, pred_poses[:, 6:12])

                            tgt_src_cloud3_4 = transform_cloud(tgt_cloud1, pred_poses1[:, 6:12], False)


                    if  opt.match_weight > 0:
                        match_pixel21 = get_src_pixel_coords(
                            tgt_image,
                            tf.squeeze(pred_depth[0][: opt.batch_size, :, :, :], axis=3),
                            pred_poses[:, :6],
                            intrinsics[:, 0, :, :])
                        match_pixel23 = get_src_pixel_coords(
                            tgt_image,
                            tf.squeeze(pred_depth[0][: opt.batch_size, :, :, :], axis=3),
                            pred_poses[:, 6:],
                            intrinsics[:, 0, :, :])
                        match_pixel_stack = tf.concat([match_pixel21, match_pixel23], axis=3)
                        match_pixel32 = get_src_pixel_coords(
                            tgt_image1,
                            tf.squeeze(pred_depth1[0][: opt.batch_size, :, :, :], axis=3),
                            pred_poses1[:, :6],
                            intrinsics[:, 0, :, :])
                        match_pixel34 = get_src_pixel_coords(
                            tgt_image1,
                            tf.squeeze(pred_depth1[0][: opt.batch_size, :, :, :], axis=3),
                            pred_poses1[:, 6:],
                            intrinsics[:, 0, :, :])
                        match_pixel_stack1 = tf.concat([match_pixel32, match_pixel34], axis=3)


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
                            match_error_map = match_pixel_stack[:, :, :, i * 1:(i + 1) * 1] * \
                                              matchmask[:, :, :, i * 1:(i + 1) * 1] - match[:, :, :, i * 2:(i + 1) * 2]
                            pixel_match_loss += tf.reduce_mean(tf.abs(match_error_map) * mask)

                        if opt.epipolar_loss_weight > 0:
                            if i == 0:
                                tgt_src_cloud = tgt_src_cloud2_1
                                essential_mat = essentialmat[:, 0, :, :]
                            else:
                                tgt_src_cloud = tgt_src_cloud2_3
                                essential_mat = essential_mat_from_rt(pred_poses[:, 6:], intrinsics[:, s, :, :])

                            epipolar_loss += self.compute_epipolar_error(
                                tgt_cloud,
                                tgt_src_cloud,
                                essential_mat,
                                mask)

                        if i == 1:
                            if opt.depth_weight > 0:
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
                        # below-threshold mask
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
                        if opt.match_weight:
                            match_error_map = match_pixel_stack1[:, :, :, i * 1:(i + 1) * 1] * \
                                              matchmask[:, :, :, i * 1 + 2:(i + 1) * 1 + 2] - \
                                              match[:, :, :, i * 2 + 4:(i + 1) * 2 + 4]
                            pixel_match_loss += tf.reduce_mean(tf.abs(match_error_map) * mask)

                        if opt.epipolar_loss_weight > 0:
                            if i == 1:
                                tgt_src_cloud1 = tgt_src_cloud3_4
                                essential_mat = essentialmat[:, 3, :, :]
                            else:
                                tgt_src_cloud1 = tgt_src_cloud3_2
                                essential_mat = essential_mat_from_rt(pred_poses1[:, :6], intrinsics[:, s, :, :])

                            epipolar_loss += self.compute_epipolar_error(
                                tgt_cloud1,
                                tgt_src_cloud1,
                                essential_mat,
                                mask)

                        if i == 0:
                            if opt.depth_weight > 0:
                                depth_loss += tf.reduce_mean(
                                    self.compute_depth_loss(curr_proj_depth, computed_depth) * mask)

                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.cm_mask:
                            exp_mask_stack = mask
                    else:
                        # proj_depth_stack = curr_proj_depth
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

            # match_loss weight
            incr_xs = [1.0, 0.0]
            bound_incrx = [np.int(self.total_step * 2 / 5)]
            self.x = tf.train.piecewise_constant(self.global_step, bound_incrx, incr_xs)

            #lr
            learning_rates = [opt.start_learning_rate, opt.start_learning_rate/10]
            boundaries = [np.int(self.total_step * 4 / 5)]
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, learning_rates)

            # boundaries = [np.int(self.total_step * 3 / 5), np.int(self.total_step * 4 / 5)]
            # values = [opt.start_learning_rate, opt.start_learning_rate / 2, opt.start_learning_rate / 10]
            # self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, opt.beta1)
            self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

            depth_loss = depth_loss * opt.depth_weight
            epipolar_loss = epipolar_loss * opt.epipolar_loss_weight
            pixel_match_loss = pixel_match_loss / (4) * opt.match_weight * self.x
            pixel_losses /= opt.num_scales

            total_loss = pixel_losses + \
                         smooth_loss + depth_loss + epipolar_loss + pixel_match_loss

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

    def compute_epipolar_error(self, p1, p2, essential_matrix, mask):
        # r(p1->p2)
        mask = tf.transpose(mask, [0, 3, 1, 2])
        p1 = mask * p1
        p1 = tf.reshape(p1, [self.opt.batch_size, 4, -1])
        p1 = tf.expand_dims(tf.transpose(p1, [0, 2, 1]), axis=-1)

        essential_matrix = tf.tile(
            tf.expand_dims(essential_matrix, axis=1),
            [1, self.opt.img_height * self.opt.img_width, 1, 1])

        ep1 = tf.matmul(essential_matrix, p1[:, :, :-1, :])
        p2 = tf.reshape(p2, [self.opt.batch_size, 4, -1])
        p2 = tf.expand_dims(tf.transpose(p2, [0, 2, 1]), axis=2)
        epipolar_error = tf.matmul(p2[:, :, :, :-1], ep1)
        return tf.reduce_mean(tf.abs(epipolar_error))


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
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

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
                                    max_to_keep=10)
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
                fetches["x"] = self.x
                # fetches["incr_x"] = self.incr_x
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
