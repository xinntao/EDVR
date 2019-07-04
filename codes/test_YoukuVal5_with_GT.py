'''
test YoukuVal5 (SR-blur) datasets
write to txt log file
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import utils.util as util
import data.util as data_util
import models.modules.EDVR_arch as EDVR_arch


def main():
    #################
    # configurations
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    data_mode = 'Val5'  # Val5 | Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
    # Vid4: SR
    # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
    #        blur (deblur-clean), blur_comp (deblur-compression).

    #### model
    if data_mode == 'Val5':
        model_path = '../experiments/pretrained_models/EDVRwTSA_youku.pth'
    elif data_mode == 'Vid4':
        model_path = '../experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'
    elif data_mode == 'sharp_bicubic':
        model_path = '../experiments/pretrained_models/EDVRwTSA_600000_G.pth' # EDVR_REDS_SR_L.pth
    elif data_mode == 'blur_bicubic':
        model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_L.pth'
    elif data_mode == 'blur':
        model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_L.pth'
    elif data_mode == 'blur_comp':
        model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_L.pth'
    else:
        raise NotImplementedError
    if data_mode == 'Vid4':
        N_in = 7  # use N_in images to restore one HR image
    else:
        N_in = 5
    predeblur, HR_in, w_TSA = False, False, True
    if data_mode == 'blur_bicubic':
        predeblur = True
    if data_mode == 'blur' or data_mode == 'blur_comp':
        predeblur, HR_in = True, True
    model = EDVR_arch.EDVR(64, N_in, 8, 5, 10, predeblur=predeblur, HR_in=HR_in, w_TSA=w_TSA) # nf: 128 and back_RBs:40 in paper, and 64 and 10 in option files

    #### dataset
    if data_mode == 'Val5':
        test_dataset_folder = '/mnt/dataset/youku_data/round1_val5_input_frames/'
    elif data_mode == 'Vid4':
        test_dataset_folder = '../datasets/Vid4/BIx4/*'
    else:
        test_dataset_folder = '../datasets/REDS4/{}/*'.format(data_mode)

    #### evaluation
    flip_test = True
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'Val5' or data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = False
    ############################################################################
    device = torch.device('cuda')
    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))

    def read_image(img_path):
        '''read one image from img_path
        Return img: HWC, BGR, [0,1], numpy
        '''
        img_GT = cv2.imread(img_path)
        img = img_GT.astype(np.float32) / 255.
        return img

    def read_seq_imgs(img_seq_path):
        '''read a sequence of images'''
        img_path_l = sorted(glob.glob(img_seq_path + '/*'))
        img_l = [read_image(v) for v in img_path_l]
        # stack to TCHW, RGB, [0,1], torch
        imgs = np.stack(img_l, axis=0)
        imgs = imgs[:, :, :, [2, 1, 0]]
        imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
        return imgs

    def index_generation(crt_i, max_n, N, padding='reflection'):
        '''
        padding: replicate | reflection | new_info | circle
        '''
        max_n = max_n - 1
        n_pad = N // 2
        return_l = []

        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            if i < 0:
                if padding == 'replicate':
                    add_idx = 0
                elif padding == 'reflection':
                    add_idx = -i
                elif padding == 'new_info':
                    add_idx = (crt_i + n_pad) + (-i)
                elif padding == 'circle':
                    add_idx = N + i
                else:
                    raise ValueError('Wrong padding mode')
            elif i > max_n:
                if padding == 'replicate':
                    add_idx = max_n
                elif padding == 'reflection':
                    add_idx = max_n * 2 - i
                elif padding == 'new_info':
                    add_idx = (crt_i - n_pad) - (i - max_n)
                elif padding == 'circle':
                    add_idx = i - N
                else:
                    raise ValueError('Wrong padding mode')
            else:
                add_idx = i
            return_l.append(add_idx)
        return return_l

    def single_forward(model, imgs_in):
        with torch.no_grad():
            if data_mode == 'Val5':
                padding_h, padding_w = imgs_in.size(-2) % 4 // 2, imgs_in.size(-1) % 4 //2
                imgs_in = F.pad(imgs_in, pad=(padding_w, padding_w, padding_h, padding_h))
            model_output = model(imgs_in)
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
            
            if data_mode == 'Val5':
                if padding_w == 0:
                    output = output[:, :, padding_h*4:-padding_h*4, :]
                elif padding_h == 0:
                    output = output[:, :, :, padding_w*4:-padding_w*4]
                else:
                    output = output[:, :, padding_h*4:-padding_h*4, padding_w*4:-padding_w*4]
        return output

    if data_mode == 'Val5':
        sub_folder_l = [test_dataset_folder]
    else:
        sub_folder_l = sorted(glob.glob(test_dataset_folder))
    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    sub_folder_name_l = []

    # for each sub-folder
    for sub_folder in sub_folder_l:
        sub_folder_name = sub_folder.split('/')[-1]
        sub_folder_name_l.append(sub_folder_name)
        save_sub_folder = osp.join(save_folder, sub_folder_name)

        img_path_l = sorted(glob.glob(sub_folder + '/*'))
        if data_mode=='Val5':
            max_idx = 100
        else:
            max_idx = len(img_path_l)

        if save_imgs:
            util.mkdirs(save_sub_folder)

        #### read LR images
        imgs = read_seq_imgs(sub_folder)
        #### read GT images
        img_GT_l = []
        if data_mode == 'Val5':
            sub_folder_GT = osp.join(sub_folder.replace('input', 'label'), '*')
        elif data_mode == 'Vid4':
            sub_folder_GT = osp.join(sub_folder.replace('/BIx4/', '/GT/'), '*')
        else:
            sub_folder_GT = osp.join(sub_folder.replace('/{}/'.format(data_mode), '/GT/'), '*')
        for img_GT_path in sorted(glob.glob(sub_folder_GT)):
            # print(img_GT_path)
            img_GT_l.append(read_image(img_GT_path))

        avg_psnr, avg_psnr_border, avg_psnr_center = 0, 0, 0
        cal_n_border, cal_n_center = 0, 0
        if data_mode == 'Val5': 
            base_index = 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            if data_mode == 'Val5':
                c_idx = int(osp.basename(img_path)[-7:-4]) - 1
            else:
                c_idx = int(osp.splitext(osp.basename(img_path))[0])

            select_idx = index_generation(c_idx, max_idx, N_in, padding=padding)
            # print(img_path, c_idx, select_idx)
            # get input images
            
            if data_mode == 'Val5':
                if img_idx > 0 and img_idx % 100 == 0:
                    base_index += 100
                select_idx = [idx + base_index for idx in  select_idx]

            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            output = single_forward(model, imgs_in)
            output_f = output.data.float().cpu().squeeze(0)

            if flip_test:
                # flip W
                output = single_forward(model, torch.flip(imgs_in, (-1, )))
                output = torch.flip(output, (-1, ))
                output = output.data.float().cpu().squeeze(0)
                output_f = output_f + output
                # flip H
                output = single_forward(model, torch.flip(imgs_in, (-2, )))
                output = torch.flip(output, (-2, ))
                output = output.data.float().cpu().squeeze(0)
                output_f = output_f + output
                # flip both H and W
                output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                output = torch.flip(output, (-2, -1))
                output = output.data.float().cpu().squeeze(0)
                output_f = output_f + output

                output_f = output_f / 4

            output = util.tensor2img(output_f)

            # save imgs
            if save_imgs:
                cv2.imwrite(osp.join(save_sub_folder, '{:08d}.png'.format(c_idx)), output)

            #### calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            # For REDS, evaluate on RGB channels; for Vid4 and Val5, evaluate on Y channels
            if data_mode == 'Vid4' or data_mode == 'Val5':  # bgr2y, [0, 1]
                GT = data_util.bgr2ycbcr(GT)
                output = data_util.bgr2ycbcr(output)
            if crop_border == 0:
                cropped_output = output
                cropped_GT = GT
            else:
                cropped_output = output[crop_border:-crop_border, crop_border:-crop_border]
                cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border]
            crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
            logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB'.format(img_idx + 1, c_idx, crt_psnr))

            if (img_idx >= border_frame and img_idx < max_idx - border_frame) or \
                (img_idx % 100 >= border_frame and img_idx % 100 < max_idx - border_frame):  # center frames
                avg_psnr_center += crt_psnr
                cal_n_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                cal_n_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (cal_n_center + cal_n_border)
        avg_psnr_center = avg_psnr_center / cal_n_center
        if cal_n_border == 0:
            avg_psnr_border = 0
        else:
            avg_psnr_border = avg_psnr_border / cal_n_border

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(sub_folder_name, avg_psnr,
                                                                   (cal_n_center + cal_n_border),
                                                                   avg_psnr_center, cal_n_center,
                                                                   avg_psnr_border, cal_n_border))

        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

    logger.info('################ Tidy Outputs ################')
    for name, psnr, psnr_center, psnr_border in zip(sub_folder_name_l, avg_psnr_l,
                                                    avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'.format(name, psnr, psnr_center, psnr_border))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(sub_folder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l)))


if __name__ == '__main__':
    main()
