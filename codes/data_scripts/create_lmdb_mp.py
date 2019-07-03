'''create lmdb files for Vimeo90K / REDS training dataset (multiprocessing)
Will read all the images to the memory'''

import sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2
try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    import data.util as data_util
    import utils.util as util
except ImportError:
    pass


def reading_image_worker(path, key):
    '''worker for reading images'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def vimeo90k():
    '''create lmdb for the Vimeo90K dataset, each image with fixed size
    GT: [3, 256, 448]
        Only need the 4th frame currently, e.g., 00001_0001_4
    LR: [3, 64, 112]
        With 1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001
    '''
    #### configurations
    mode = 'GT'  # GT | LR
    read_all_imgs = False  # whether real all images to the memory. Set False with limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    if mode == 'GT':
        img_folder = '/home/xtwang/datasets/vimeo90k/vimeo_septuplet/sequences'
        lmdb_save_path = '/home/xtwang/datasets/vimeo90k/vimeo90k_train_GT.lmdb'
        txt_file = '/home/xtwang/datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 256, 448
    elif mode == 'LR':
        img_folder = '/home/xtwang/datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
        lmdb_save_path = '/home/xtwang/datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
        txt_file = '/home/xtwang/datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 64, 112
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        file_l = glob.glob(osp.join(img_folder, folder, sub_folder) + '/*')
        all_img_list.extend(file_l)
        for j in range(7):
            keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'GT':  # read the 4th frame only for GT mode
        print('Only keep the 4th frame.')
        all_img_list = [v for v in all_img_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('_4')]

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(reading_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    idx = 1
    for path, key in zip(all_img_list, keys):
        idx = idx + 1
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        H, W, C = data.shape  # fixed shape
        assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 1:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    if mode == 'GT':
        meta_info['name'] = 'Vimeo90K_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'Vimeo90K_train_LR'
    meta_info['resolution'] = '{}_{}_{}'.format(3, H_dst, W_dst)
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = key_set
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def REDS():
    '''create lmdb for the REDS dataset, each image with fixed size
    GT: [3, 720, 1280], key: 000_00000000
    LR: [3, 180, 320], key: 000_00000000
    key: 000_00000000
    '''
    #### configurations
    mode = 'train_sharp'
    read_all_imgs = False  # whether real all images to the memory. Set False with limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    # train_sharp | train_sharp_bicubic | train_blur_bicubic| train_blur | train_blur_comp
    if mode == 'train_sharp':
        img_folder = '/home/xtwang/datasets/REDS/train_sharp'
        lmdb_save_path = '/home/xtwang/datasets/REDS/train_sharp_wval.lmdb'
        H_dst, W_dst = 720, 1280
    elif mode == 'train_sharp_bicubic':
        img_folder = '/home/xtwang/datasets/REDS/train_sharp_bicubic'
        lmdb_save_path = '/home/xtwang/datasets/REDS/train_sharp_bicubic_wval.lmdb'
        H_dst, W_dst = 180, 320
    elif mode == 'train_blur_bicubic':
        img_folder = '/home/xtwang/datasets/REDS/train_blur_bicubic'
        lmdb_save_path = '/home/xtwang/datasets/REDS/train_blur_bicubic_wval.lmdb'
        H_dst, W_dst = 180, 320
    elif mode == 'train_blur':
        img_folder = '/home/xtwang/datasets/REDS/train_blur'
        lmdb_save_path = '/home/xtwang/datasets/REDS/train_blur_wval.lmdb'
        H_dst, W_dst = 720, 1280
    elif mode == 'train_blur_comp':
        img_folder = '/home/xtwang/datasets/REDS/train_blur_comp'
        lmdb_save_path = '/home/xtwang/datasets/REDS/train_blur_comp_wval.lmdb'
        H_dst, W_dst = 720, 1280
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = data_util._get_paths_from_images(img_folder)
    keys = []
    for img_path in all_img_list:
        split_rlt = img_path.split('/')
        a = split_rlt[-2]
        b = split_rlt[-1].split('.png')[0]
        keys.append(a + '_' + b)

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(reading_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    if 'flow' in mode:
        data_size_per_img = dataset['000_00000002_n1'].nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    idx = 1
    for path, key in zip(all_img_list, keys):
        idx = idx + 1
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if 'flow' in mode:
            H, W = data.shape
            assert H == H_dst and W == W_dst, 'different shape.'
        else:
            H, W, C = data.shape  # fixed shape
            assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 1:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    meta_info['name'] = 'REDS_{}_wval'.format(mode)
    if 'flow' in mode:
        meta_info['resolution'] = '{}_{}_{}'.format(1, H_dst, W_dst)
    else:
        meta_info['resolution'] = '{}_{}_{}'.format(3, H_dst, W_dst)
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def test_lmdb(dataroot, dataset='REDS'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    # read one image
    if dataset == 'vimeo90k':
        key = '00001_0001_4'
    else:
        key = '000_00000000'
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
    img = img_flat.reshape(H, W, C)
    cv2.imwrite('test.png', img)


if __name__ == "__main__":
    # vimeo90k()
    REDS()
    # test_lmdb('/home/xtwang/datasets/REDS/train_sharp_wval.lmdb', 'REDS')
