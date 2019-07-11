'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util

logger = logging.getLogger('base')


class YoukuTrain(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Resolution, e.g., LR video frames
    support reading N LR frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(YoukuTrain, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.BIX2_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_BIX2'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        if opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            cache_keys = opt['cache_keys']
        else:
            cache_keys = 'Youku_train_keys.pkl'
        logger.info('Using cache keys - {}.'.format(cache_keys))
        self.paths_GT = pickle.load(open('./data/{}'.format(cache_keys), 'rb'))
        assert self.paths_GT, 'Error: GT path is empty.'
        # load data resolution
        meta = pickle.load(open(osp.join(self.GT_root, 'meta_info.pkl'), 'rb'))
        self.resolutions_GT = meta['resolution']
        meta = pickle.load(open(osp.join(self.BIX2_root, 'meta_info.pkl'), 'rb'))
        self.resolutions_BIX2 = meta['resolution']
        meta = pickle.load(open(osp.join(self.LQ_root, 'meta_info.pkl'), 'rb'))
        self.resolutions_LQ = meta['resolution']

        if self.data_type == 'lmdb':
            self.GT_env, self.BIX2_env, self.LQ_env = None, None, None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.BIX2_env = lmdb.open(self.opt['dataroot_BIX2'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        # if self.data_type == 'lmdb':
        if (self.GT_env is None) or (self.BIX2_env) is None or (self.LQ_env is None):
            self._init_lmdb()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 100:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 1:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:03d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >
                   100) or (center_frame_idx - self.half_N_frames * interval < 1):
                center_frame_idx = random.randint(1, 100)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:03d}'.format(neighbor_list[self.half_N_frames])

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))

        #### get the GT image (as the center frame)
        resolutions_GT = [int(s) for s in self.resolutions_GT[name_a].split('_')]
        resolutions_BIX2 = [int(s) for s in self.resolutions_BIX2[name_a].split('_')]
        if self.data_type == 'lmdb':
            img_GT = util.read_img(self.GT_env, key, resolutions_GT)
            img_BIX2 = util.read_img(self.BIX2_env, key, resolutions_BIX2) 
        else:
            # img_GT filename is Youku_00059_h_GT100.bmp
            img_GT = util.read_img(None, osp.join(self.GT_root, 'Youku_{}_h_GT{}.bmp'.format(name_a, name_b))) 
            # the names of BIX2 imgs are the same as GT.
            img_BIX2 = util.read_img(None, osp.join(self.BIX2_root, 'Youku_{}_h_GT{}.bmp'.format(name_a, name_b)))  

        #### get LQ images
        
        resolutions_LQ = [int(s) for s in self.resolutions_LQ[name_a].split('_')]
        LQ_size_tuple = resolutions_LQ if self.LR_input else resolutions_GT
        img_LQ_l = []
        for v in neighbor_list:
            # img_LQ filename is Youku_00059_l100.bmp
            img_LQ_path = osp.join(self.LQ_root, 'Youku_{}_l{:03d}.bmp'.format(name_a, v))
            if self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, '{}_{:03d}'.format(name_a, v), LQ_size_tuple) 
            else:
                img_LQ = util.read_img(None, img_LQ_path)
            img_LQ_l.append(img_LQ)            

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            # if self.LR_input:
            LQ_size = GT_size // scale
            BIX2_size = GT_size // 2
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
            rnd_h_BIX2, rnd_w_BIX2 = int(rnd_h * 2), int(rnd_w * 2) # for 2times upscale
            img_BIX2 = img_BIX2[rnd_h_BIX2:rnd_h_BIX2 + BIX2_size, rnd_w_BIX2:rnd_w_BIX2 + BIX2_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            # else:
            #     rnd_h = random.randint(0, max(0, H - GT_size))
            #     rnd_w = random.randint(0, max(0, W - GT_size))
            #     img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
            #     img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_BIX2)
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-2]
            img_BIX2 = rlt[-2]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_BIX2 = img_BIX2[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_BIX2s = torch.from_numpy(np.ascontiguousarray(np.transpose(img_BIX2, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'BIX2s': img_BIX2s,'GT': img_GT, 'key': key}

    def __len__(self):
        return len(self.paths_GT)