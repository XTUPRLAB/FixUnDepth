# -*- coding: utf-8 -*-
from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np

class DataLoader(object):
    def __init__(self,
                 dataset_dir=None,
                 batch_size=None,
                 img_height=None,
                 img_width=None,
                 num_source=None,
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train')
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'],
            seed=seed,
            shuffle=True)
        cam_paths_queue = tf.train.string_input_producer(
            file_list['cam_file_list'],
            seed=seed,
            shuffle=True)
        em_paths_queue = tf.train.string_input_producer(
            file_list['em_file_list'],
            seed=seed,
            shuffle=True)
        emtr_paths_queue = tf.train.string_input_producer(
            file_list['emtr_file_list'],
            seed=seed,
            shuffle=True)
        mp1_paths_queue = tf.train.string_input_producer(
            file_list['mp1_file_list'],
            seed=seed,
            shuffle=True)
        mp2_paths_queue = tf.train.string_input_producer(
            file_list['mp2_file_list'],
            seed=seed,
            shuffle=True)
        mp3_paths_queue = tf.train.string_input_producer(
           file_list['mp3_file_list'],
           seed=seed,
           shuffle=True)
        mp4_paths_queue = tf.train.string_input_producer(
           file_list['mp4_file_list'],
           seed=seed,
           shuffle=True)
        mp1tr_paths_queue = tf.train.string_input_producer(
            file_list['mp1tr_file_list'],
            seed=seed,
            shuffle=True)
        mp2tr_paths_queue = tf.train.string_input_producer(
            file_list['mp2tr_file_list'],
            seed=seed,
            shuffle=True)
        mp3tr_paths_queue = tf.train.string_input_producer(
           file_list['mp3tr_file_list'],
           seed=seed,
           shuffle=True)
        mp4tr_paths_queue = tf.train.string_input_producer(
           file_list['mp4tr_file_list'],
           seed=seed,
           shuffle=True)
        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)
        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image, tgt_image1,src_image_stack,src_image_stack1 = \
            self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)


        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])

        # Load essential matrix
        essentialmat = self.load_ess_matrix(em_paths_queue)
        essentialmat_tran = self.load_ess_matrix(emtr_paths_queue)

        num = 50
        # Load match points
        
        match_points1 = self.load_match_points(mp1_paths_queue)
        match_points2 = self.load_match_points(mp2_paths_queue)
        match_points3 = self.load_match_points(mp3_paths_queue)
        match_points4 = self.load_match_points(mp4_paths_queue)

        
        match_points_stack = tf.concat([match_points1, match_points2, match_points3, match_points4], axis=-1)

        # Load match_trans points
        match_points1_tran = self.load_match_points(mp1tr_paths_queue)
        match_points2_tran = self.load_match_points(mp2tr_paths_queue)
        match_points3_tran = self.load_match_points(mp3tr_paths_queue)
        match_points4_tran = self.load_match_points(mp4tr_paths_queue)
        match_points_tran_stack = tf.concat([match_points1_tran, match_points2_tran, match_points3_tran, match_points4_tran], axis=-1)

        # Form training batches
        src_image_stack, tgt_image,src_image_stack1, tgt_image1, intrinsics, essentialmat,  essentialmat_tran, match_points_stack, match_points_tran_stack = \
                tf.train.batch([src_image_stack, tgt_image, src_image_stack1, tgt_image1,intrinsics, essentialmat,  essentialmat_tran, match_points_stack, match_points_tran_stack],
                               batch_size=self.batch_size)

        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack, tgt_image1, src_image_stack1], axis=3)
        image_all_aug = image_all
        image_all, image_all_aug, match_points_stack, essentialmat, flag = self.data_augmentation(
            image_all, match_points_stack, match_points_tran_stack, essentialmat, essentialmat_tran)

        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:9]
        tgt_image1 = image_all[:, :, :, 9:12]
        src_image_stack1 = image_all[:, :, :, 12:]

        tgt_image_aug = image_all_aug[:, :, :, :3]
        src_image_stack_aug = image_all_aug[:, :, :, 3:9]
        tgt_image_aug1 = image_all_aug[:, :, :, 9:12]
        src_image_stack_aug1 = image_all_aug[:, :, :, 12:]
        

        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        return  tgt_image, src_image_stack, tgt_image1, src_image_stack1, tgt_image_aug, src_image_stack_aug, \
                tgt_image_aug1, src_image_stack_aug1, intrinsics, essentialmat, match_points_stack, flag

    def load_ess_matrix(self, em_paths_queue):
        em_reader = tf.TextLineReader()
        _, raw_em_contents = em_reader.read(em_paths_queue)
        rec_em_def = []
        for i in range(36):
            rec_em_def.append([1.])
        raw_em_vec = tf.decode_csv(raw_em_contents,
                                    record_defaults=rec_em_def)
        raw_em_vec = tf.stack(raw_em_vec)
        essentialmat = tf.reshape(raw_em_vec, [4, 3, 3])

        return essentialmat

    def load_match_points(self,paths_queue, num=50):
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(paths_queue)
        rec_def = []
        for i in range(num * 4):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        match_points = tf.reshape(raw_cam_vec, [num, 4])
        return match_points

    def unpack_match_sequence(self, match_seq, img_height, img_width):
        match_seq = tf.slice(match_seq, [0, 0, 0], [-1, -1, 2])
        match21 = match_seq[:, :img_width, :]
        match23 = match_seq[:, img_width: 2 * img_width, :]
        match32 = match_seq[:, 2 * img_width:3 * img_width, :]
        match34 = match_seq[:, 3 * img_width:4 * img_width, :]
        match = tf.concat([match21, match23, match32, match34], axis=-1)
        match = tf.cast(match, tf.float32)
        match.set_shape([img_height, img_width, 8])
        return match

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics


    # add random brightness, contrast, saturation and hue to all source image
    # [H, W, (num_source + 1) * 3]
    def data_augmentation(self, im, mt, mttran, ess, esstran):
        def random_flip(im, mt, mttran, ess, esstran):
            # def flip_one(sim):
            #     do_flip = tf.random_uniform([], 0, 1)
            #     return tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(sim), lambda: sim)
            # im = tf.map_fn(lambda sim: flip_one(sim), im)
            def f1():
                return im,mt,ess
            def f2():
                return tf.image.flip_left_right(im), mttran, esstran
            # do_flip = tf.random_uniform([], 0, 1)
            do_flip = tf.Variable(tf.random_uniform([], 0, 1), name='do_flip', trainable=False, dtype=tf.float32)
            flag = tf.abs(tf.assign(do_flip, tf.random_uniform([], 0, 1)))

            im, mt, ess = tf.cond(do_flip > 0.5, f2, f1)
            
            return im, mt, ess, flag

        def augment_image_properties(im):
            # random brightness
            brightness_seed = random.randint(0, 2 ** 31 - 1)
            im = tf.image.random_brightness(im, 0.2, brightness_seed)

            contrast_seed = random.randint(0, 2 ** 31 - 1)
            im = tf.image.random_contrast(im, 0.8, 1.2, contrast_seed)

            num_img = np.int(im.get_shape().as_list()[-1] // 3)

            # saturation_seed = random.randint(0, 2**31 - 1)
            saturation_im_list = []
            saturation_factor = random.uniform(0.8,
                                               1.2)  # tf.random_ops.random_uniform([], 0.8, 1.2, seed=saturation_seed)
            for i in range(num_img):
                saturation_im_list.append(
                    tf.image.adjust_saturation(im[:, :, 3 * i: 3 * (i + 1)], saturation_factor))
                # tf.image.random_saturation(im[:,:, 3*i: 3*(i+1)], 0.8, 1.2, seed=saturation_seed))
            im = tf.concat(saturation_im_list, axis=2)

            # hue_seed = random.randint(0, 2 ** 31 - 1)
            hue_im_list = []
            hue_delta = random.uniform(-0.1, 0.1)  # tf.random_ops.random_uniform([], -0.1, 0.1, seed=hue_seed)
            for i in range(num_img):
                hue_im_list.append(tf.image.adjust_hue(im[:, :, 3 * i: 3 * (i + 1)], hue_delta))
                #  tf.image.random_hue(im[:, :, 3 * i: 3 * (i + 1)], 0.1, seed=hue_seed))
            im = tf.concat(hue_im_list, axis=2)
            return im

        def random_augmentation(im):
            def augmentation_one(sim):
                do_aug = tf.random_uniform([], 0, 1)
                return tf.cond(do_aug > 0.5, lambda: augment_image_properties(sim), lambda: sim)

            im = tf.map_fn(lambda sim: augmentation_one(sim), im)
            # im = tf.cond(do_aug > 0.5, lambda: tf.map_fn(lambda sim: augment_image_properties(sim), im), lambda: im)
            return im

        im, mt, ess, flag = random_flip(im, mt, mttran, ess, esstran)
        im_aug = random_augmentation(im)
        return im, im_aug, mt, ess, flag

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]

        em_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_em.txt') for i in range(len(frames))]

        emtr_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_em_tran.txt') for i in range(len(frames))]

        mp1_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_match1.txt') for i in range(len(frames))]
        mp2_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_match2.txt') for i in range(len(frames))]
        mp3_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_match3.txt') for i in range(len(frames))]
        mp4_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_match4.txt') for i in range(len(frames))]

        mp1tr_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_match1_tran.txt') for i in range(len(frames))]
        mp2tr_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_match2_tran.txt') for i in range(len(frames))]
        mp3tr_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_match3_tran.txt') for i in range(len(frames))]
        mp4tr_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_match4_tran.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list

        all_list['em_file_list'] = em_file_list
        all_list['emtr_file_list'] = emtr_file_list

        all_list['mp1_file_list'] = mp1_file_list
        all_list['mp2_file_list'] = mp2_file_list
        all_list['mp3_file_list'] = mp3_file_list
        all_list['mp4_file_list'] = mp4_file_list

        all_list['mp1tr_file_list'] = mp1tr_file_list
        all_list['mp2tr_file_list'] = mp2tr_file_list
        all_list['mp3tr_file_list'] = mp3tr_file_list
        all_list['mp4tr_file_list'] = mp4tr_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        tgt_image1 = tf.slice(image_seq,
                              [0, int(tgt_start_idx + img_width), 0],
                              [-1, int(img_width * (num_source // 2)), -1])
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width + img_width), 0],
                               [-1, int(img_width * (num_source // 2)), -1])
        src_image_seq = tf.concat([src_image_1, tgt_image1], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        src_image_seq1 = tf.concat([tgt_image, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack1 = tf.concat([tf.slice(src_image_seq1,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack1.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        tgt_image1.set_shape([img_height, img_width, 3])
        return tgt_image, tgt_image1, src_image_stack, src_image_stack1


    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, 0, tgt_start_idx, 0],
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0, 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, 0, int(tgt_start_idx + img_width), 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, 0, i*img_width, 0],
                                    [-1, -1, img_width, -1])
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
