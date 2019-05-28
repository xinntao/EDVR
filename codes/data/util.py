'''utils for dataloader'''
import os
import pickle
import random
import logging
import numpy as np
import lmdb
import cv2

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb cache keys'''
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')
    logger = logging.getLogger('base')
    if os.path.isfile(keys_cache_file):
        logger.info('Read lmdb keys from cache: {}'.format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            logger.info('Creating lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths


def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


###################### read images ######################
def _read_lmdb_img(env, key, size=None):
    '''read image from lmdb with key (w/ and w/o fixed size)
    if size: a (H, W, C) tuple: fixed size
    else: read size form the corresponding .meta key'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
        if not size:
            buf_meta = txn.get((key + '.meta').encode('ascii')).decode('ascii')
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    if size:
        H, W, C = size
    else:
        H, W, C = [int(s) for s in buf_meta.split(',')]
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def augment_flow(img_list, flow_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = flow[:, ::-1, :]
            flow[:, :, 0] *= -1
        if vflip:
            flow = flow[::-1, :, :]
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]

    return rlt_img_list, rlt_flow_list


def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img
