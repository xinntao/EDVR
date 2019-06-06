'''Network architecture for DUF:
Deep Video Super-Resolution Network Using Dynamic Upsampling Filters
Without Explicit Motion Compensation (CVPR18)
'''
'''
For all the models below, [adapt_official] is only necessary when
loading the weights converted from the official TensorFlow weights.
Please set it to [False] if you are training the model from scratch.
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def _initialize_weights(net, scale=1):
    '''initialize'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # init.constant_(m.weight, 1)
            m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
            init.constant_(m.bias, 0)
        # 3D
        elif isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)


class DenseBlock(nn.Module):
    '''Dense block w/ BN
    for the second denseblock, t_reduced = True'''

    def __init__(self, nf=64, ng=32, t_reduce=False):
        super(DenseBlock, self).__init__()
        self.t_reduce = t_reduce
        if self.t_reduce:
            pad = (0, 1, 1)
        else:
            pad = (1, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True)

        self.bn3d_3 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + ng,
                                  nf + ng, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=True)
        self.bn3d_4 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True)

        self.bn3d_5 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_5 = nn.Conv3d(nf + 2 * ng,
                                  nf + 2 * ng, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=True)
        self.bn3d_6 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_6 = nn.Conv3d(nf + 2 * ng,
                                  ng, (3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=pad,
                                  bias=True)

        # initialization
        _initialize_weights(self.bn3d_1, 0.1)
        _initialize_weights(self.conv3d_1, 0.1)
        _initialize_weights(self.bn3d_2, 0.1)
        _initialize_weights(self.conv3d_2, 0.1)
        _initialize_weights(self.bn3d_3, 0.1)
        _initialize_weights(self.conv3d_3, 0.1)
        _initialize_weights(self.bn3d_4, 0.1)
        _initialize_weights(self.conv3d_4, 0.1)
        _initialize_weights(self.bn3d_5, 0.1)
        _initialize_weights(self.conv3d_5, 0.1)
        _initialize_weights(self.bn3d_6, 0.1)
        _initialize_weights(self.conv3d_6, 0.1)

    def forward(self, x):
        '''x: [B, C, T, H, W]
        C: 1) 64 -> 160; 2) 160 -> 256
        T: 1) 7 -> 7; 2) 7 -> 7 - 2 * 3 = 1 (t_reduce=True)'''
        x1 = self.conv3d_1(F.relu(self.bn3d_1(x), inplace=True))
        x1 = self.conv3d_2(F.relu(self.bn3d_2(x1), inplace=True))
        if self.t_reduce:
            x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)
        else:
            x1 = torch.cat((x, x1), 1)

        x2 = self.conv3d_3(F.relu(self.bn3d_3(x1), inplace=True))
        x2 = self.conv3d_4(F.relu(self.bn3d_4(x2), inplace=True))
        if self.t_reduce:
            x2 = torch.cat((x1[:, :, 1:-1, :, :], x2), 1)
        else:
            x2 = torch.cat((x1, x2), 1)

        x3 = self.conv3d_5(F.relu(self.bn3d_5(x2), inplace=True))
        x3 = self.conv3d_6(F.relu(self.bn3d_6(x3), inplace=True))
        if self.t_reduce:
            x3 = torch.cat((x2[:, :, 1:-1, :, :], x3), 1)
        else:
            x3 = torch.cat((x2, x3), 1)
        return x3


class DynamicUpsamplingFilter_3C(nn.Module):
    '''dynamic upsampling filter with 3 channels applying the same filters
    filter_size: filter size of the generated filters, shape (C, kH, kW)'''

    def __init__(self, filter_size=(1, 5, 5), device=torch.device('cuda')):
        super(DynamicUpsamplingFilter_3C, self).__init__()
        # generate a local expansion filter, used similar to im2col
        nF = np.prod(filter_size)
        expand_filter_np = np.reshape(np.eye(nF, nF),
                                      (nF, filter_size[0], filter_size[1], filter_size[2]))
        expand_filter = torch.from_numpy(expand_filter_np).float()
        self.expand_filter = torch.cat((expand_filter, expand_filter, expand_filter),
                                       0)  # [75, 1, 5, 5]
        self.device = device

    def forward(self, x, filters):
        '''x: input image, [B, 3, H, W]
        filters: generate dynamic filters, [B, F, R, H, W], e.g., [B, 25, 16, H, W]
            F: prod of filter kernel size, e.g., 5*5 = 25
            R: used for upsampling, similar to pixel shuffle, e.g., 4*4 = 16 for x4
        Return: filtered image, [B, 3*R, H, W]
        '''
        B, nF, R, H, W = filters.size()
        # using group convolution
        input_expand = F.conv2d(x, self.expand_filter.to(self.device), padding=2,
                                groups=3)  # [B, 75, H, W] similar to im2col
        input_expand = input_expand.view(B, 3, nF, H, W).permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 25]
        filters = filters.permute(0, 3, 4, 1, 2)  # [B, H, W, 25, 16]
        out = torch.matmul(input_expand, filters)  # [B, H, W, 3, 16]
        return out.permute(0, 3, 4, 1, 2).view(B, 3 * R, H, W)  # [B, 3*16, H, W]


class DUF_16L(nn.Module):
    ''' same structure as that in the paper
    https://github.com/yhjo09/VSR-DUF'''

    def __init__(self, adapt_official=False):
        super(DUF_16L, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dense_block_1 = DenseBlock(64, 64 // 2, t_reduce=False)  # 64 + 32 * 3 = 160, T = 7
        self.dense_block_2 = DenseBlock(160, 64 // 2, t_reduce=True)  # 160 + 32 * 3 = 160, T = 1
        self.bn3d_2 = nn.BatchNorm3d(256, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(256,
                                  256, (1, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(0, 1, 1),
                                  bias=True)

        self.conv3d_r1 = nn.Conv3d(256,
                                   256, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_r2 = nn.Conv3d(256,
                                   3 * 16, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)

        self.conv3d_f1 = nn.Conv3d(256,
                                   512, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_f2 = nn.Conv3d(512,
                                   1 * 5 * 5 * 16, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)

        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 5, 5))

        # initialization
        _initialize_weights(self.conv3d_1, 0.1)
        _initialize_weights(self.bn3d_2, 0.1)
        _initialize_weights(self.conv3d_2, 0.1)
        _initialize_weights(self.conv3d_r1, 0.1)
        _initialize_weights(self.conv3d_r2, 0.1)
        _initialize_weights(self.conv3d_f1, 0.1)
        _initialize_weights(self.conv3d_f2, 0.1)

        self.adapt_official = adapt_official

    def forward(self, x):
        '''
        x: [B, T, C, H, W], T = 7. reshape to [B, C, T, H, W] for Conv3D
        Generate filters and image residual:
        Fx: [B, 25, 16, H, W] for DynamicUpsamplingFilter_3C
        Rx: [B, 3*16, 1, H, W]
        '''
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] for Conv3D
        x_center = x[:, :, T // 2, :, :]

        x = self.conv3d_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)  # reduce T to 1
        x = F.relu(self.conv3d_2(F.relu(self.bn3d_2(x), inplace=True)), inplace=True)

        # image residual
        Rx = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))  # [B, 3*16, 1, H, W]

        # filter
        Fx = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))  #[B, 25*16, 1, H, W]
        Fx = F.softmax(Fx.view(B, 25, 16, H, W), dim=1)

        # Adapt to official model weights
        if self.adapt_official:
            x = Rx.clone()
            x1 = x[:, ::3, :, :]
            x2 = x[:, 1::3, :, :]
            x3 = x[:, 2::3, :, :]

            Rx[:, :16, :, :] = x1
            Rx[:, 16:32, :, :] = x2
            Rx[:, 32:, :, :] = x3

        # dynamic filter
        out = self.dynamic_filter(x_center, Fx)  # [B, 3*R, H, W]
        out += Rx.squeeze_(2)
        out = F.pixel_shuffle(out, 4)  # [B, 3, H, W]

        return out


class DenseBlock_noBN(nn.Module):
    '''Dense block w/o BN
    for the second denseblock, t_reduced = True'''

    def __init__(self, nf=64, ng=32, t_reduce=False):
        super(DenseBlock_noBN, self).__init__()
        self.t_reduce = t_reduce
        if self.t_reduce:
            pad = (0, 1, 1)
        else:
            pad = (1, 1, 1)
        self.conv3d_1 = nn.Conv3d(nf,
                                  nf, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=False)
        self.conv3d_2 = nn.Conv3d(nf, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True)

        self.conv3d_3 = nn.Conv3d(nf + ng,
                                  nf + ng, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=False)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True)

        self.conv3d_5 = nn.Conv3d(nf + 2 * ng,
                                  nf + 2 * ng, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=False)
        self.conv3d_6 = nn.Conv3d(nf + 2 * ng,
                                  ng, (3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=pad,
                                  bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialization
        _initialize_weights(self.conv3d_1, 0.1)
        _initialize_weights(self.conv3d_2, 0.1)
        _initialize_weights(self.conv3d_3, 0.1)
        _initialize_weights(self.conv3d_4, 0.1)
        _initialize_weights(self.conv3d_5, 0.1)
        _initialize_weights(self.conv3d_6, 0.1)

    def forward(self, x):
        '''x: [B, C, T, H, W]
        C: 1) 64 -> 160; 2) 160 -> 256
        T: 1) 7 -> 7; 2) 7 -> 7 - 2 * 3 = 1 (t_reduce=True)'''
        x1 = self.lrelu(self.conv3d_1(x))
        x1 = self.lrelu(self.conv3d_2(x1))
        if self.t_reduce:
            x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)
        else:
            x1 = torch.cat((x, x1), 1)

        x2 = self.lrelu(self.conv3d_3(x1))
        x2 = self.lrelu(self.conv3d_4(x2))
        if self.t_reduce:
            x2 = torch.cat((x1[:, :, 1:-1, :, :], x2), 1)
        else:
            x2 = torch.cat((x1, x2), 1)

        x3 = self.lrelu(self.conv3d_5(x2))
        x3 = self.lrelu(self.conv3d_6(x3))
        if self.t_reduce:
            x3 = torch.cat((x2[:, :, 1:-1, :, :], x3), 1)
        else:
            x3 = torch.cat((x2, x3), 1)
        return x3


class DUF_16L_noBN(nn.Module):
    ''' same structure as that in the paper
    https://github.com/yhjo09/VSR-DUF'''

    def __init__(self):
        super(DUF_16L_noBN, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dense_block_1 = DenseBlock_noBN(64, 64 // 2,
                                             t_reduce=False)  # 64 + 32 * 3 = 160, T = 7
        self.dense_block_2 = DenseBlock_noBN(160, 64 // 2,
                                             t_reduce=True)  # 160 + 32 * 3 = 160, T = 1

        self.conv3d_2 = nn.Conv3d(256,
                                  256, (1, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(0, 1, 1),
                                  bias=True)

        self.conv3d_r1 = nn.Conv3d(256,
                                   256, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_r2 = nn.Conv3d(256,
                                   3 * 16, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)

        self.conv3d_f1 = nn.Conv3d(256,
                                   512, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_f2 = nn.Conv3d(512,
                                   1 * 5 * 5 * 16, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)

        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 5, 5))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # initialization
        _initialize_weights(self.conv3d_1, 0.1)
        _initialize_weights(self.conv3d_2, 0.1)
        _initialize_weights(self.conv3d_r1, 0.1)
        _initialize_weights(self.conv3d_r2, 0.1)
        _initialize_weights(self.conv3d_f1, 0.1)
        _initialize_weights(self.conv3d_f2, 0.1)

    def forward(self, x):
        '''
        x: [B, T, C, H, W], T = 7. reshape to [B, C, T, H, W] for Conv3D
        Generate filters and image residual:
        Fx: [B, 25, 16, H, W] for DynamicUpsamplingFilter_3C
        Rx: [B, 3*16, 1, H, W]
        '''
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] for Conv3D
        x_center = x[:, :, T // 2, :, :]
        x = self.conv3d_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)  # reduce T to 1
        x = self.lrelu(self.conv3d_2(x))

        # image residual
        Rx = self.conv3d_r2(self.lrelu(self.conv3d_r1(x)))  # [B, 3*16, 1, H, W]
        # filter
        Fx = self.conv3d_f2(self.lrelu(self.conv3d_f1(x)))  #[B, 25*16, 1, H, W]
        Fx = F.softmax(Fx.view(B, 25, 16, H, W), dim=1)

        # dynamic filter
        out = self.dynamic_filter(x_center, Fx)  # [B, 3*R, H, W]
        out += Rx.squeeze_(2)
        out = F.pixel_shuffle(out, 4)  # [B, 3, H, W]
        return out


class DenseBlock_28L_A(nn.Module):
    '''Dense block w/ BN
    for the second denseblock, t_reduced = True'''

    def __init__(self, nf=64, ng=16):
        super(DenseBlock_28L_A, self).__init__()
        pad = (1, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_1 = nn.Conv3d(nf, nf, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.bn3d_2 = nn.BatchNorm3d(nf, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(nf, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True)

        self.bn3d_3 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_3 = nn.Conv3d(nf + ng,
                                  nf + ng, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=True)
        self.bn3d_4 = nn.BatchNorm3d(nf + ng, eps=1e-3, momentum=1e-3)
        self.conv3d_4 = nn.Conv3d(nf + ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True)

        self.bn3d_5 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_5 = nn.Conv3d(nf + 2 * ng,
                                  nf + 2 * ng, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=True)
        self.bn3d_6 = nn.BatchNorm3d(nf + 2 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_6 = nn.Conv3d(nf + 2 * ng,
                                  ng, (3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=pad,
                                  bias=True)

        self.bn3d_7 = nn.BatchNorm3d(nf + 3 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_7 = nn.Conv3d(nf + 3 * ng,
                                  nf + 3 * ng, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=True)
        self.bn3d_8 = nn.BatchNorm3d(nf + 3 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_8 = nn.Conv3d(nf + 3 * ng,
                                  ng, (3, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=pad,
                                  bias=True)

        self.bn3d_9 = nn.BatchNorm3d(nf + 4 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_9 = nn.Conv3d(nf + 4 * ng,
                                  nf + 4 * ng, (1, 1, 1),
                                  stride=(1, 1, 1),
                                  padding=(0, 0, 0),
                                  bias=True)
        self.bn3d_10 = nn.BatchNorm3d(nf + 4 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_10 = nn.Conv3d(nf + 4 * ng,
                                   ng, (3, 3, 3),
                                   stride=(1, 1, 1),
                                   padding=pad,
                                   bias=True)

        self.bn3d_11 = nn.BatchNorm3d(nf + 5 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_11 = nn.Conv3d(nf + 5 * ng,
                                   nf + 5 * ng, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.bn3d_12 = nn.BatchNorm3d(nf + 5 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_12 = nn.Conv3d(nf + 5 * ng,
                                   ng, (3, 3, 3),
                                   stride=(1, 1, 1),
                                   padding=pad,
                                   bias=True)

        self.bn3d_13 = nn.BatchNorm3d(nf + 6 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_13 = nn.Conv3d(nf + 6 * ng,
                                   nf + 6 * ng, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.bn3d_14 = nn.BatchNorm3d(nf + 6 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_14 = nn.Conv3d(nf + 6 * ng,
                                   ng, (3, 3, 3),
                                   stride=(1, 1, 1),
                                   padding=pad,
                                   bias=True)

        self.bn3d_15 = nn.BatchNorm3d(nf + 7 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_15 = nn.Conv3d(nf + 7 * ng,
                                   nf + 7 * ng, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.bn3d_16 = nn.BatchNorm3d(nf + 7 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_16 = nn.Conv3d(nf + 7 * ng,
                                   ng, (3, 3, 3),
                                   stride=(1, 1, 1),
                                   padding=pad,
                                   bias=True)

        self.bn3d_17 = nn.BatchNorm3d(nf + 8 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_17 = nn.Conv3d(nf + 8 * ng,
                                   nf + 8 * ng, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.bn3d_18 = nn.BatchNorm3d(nf + 8 * ng, eps=1e-3, momentum=1e-3)
        self.conv3d_18 = nn.Conv3d(nf + 8 * ng,
                                   ng, (3, 3, 3),
                                   stride=(1, 1, 1),
                                   padding=pad,
                                   bias=True)

        # initialization
        _initialize_weights(self.bn3d_1, 0.1)
        _initialize_weights(self.conv3d_1, 0.1)
        _initialize_weights(self.bn3d_2, 0.1)
        _initialize_weights(self.conv3d_2, 0.1)
        _initialize_weights(self.bn3d_3, 0.1)
        _initialize_weights(self.conv3d_3, 0.1)
        _initialize_weights(self.bn3d_4, 0.1)
        _initialize_weights(self.conv3d_4, 0.1)
        _initialize_weights(self.bn3d_5, 0.1)
        _initialize_weights(self.conv3d_5, 0.1)
        _initialize_weights(self.bn3d_6, 0.1)
        _initialize_weights(self.conv3d_6, 0.1)
        _initialize_weights(self.bn3d_7, 0.1)
        _initialize_weights(self.conv3d_7, 0.1)
        _initialize_weights(self.bn3d_8, 0.1)
        _initialize_weights(self.conv3d_8, 0.1)
        _initialize_weights(self.bn3d_9, 0.1)
        _initialize_weights(self.conv3d_9, 0.1)
        _initialize_weights(self.bn3d_10, 0.1)
        _initialize_weights(self.conv3d_10, 0.1)
        _initialize_weights(self.bn3d_11, 0.1)
        _initialize_weights(self.conv3d_11, 0.1)
        _initialize_weights(self.bn3d_12, 0.1)
        _initialize_weights(self.conv3d_12, 0.1)
        _initialize_weights(self.bn3d_13, 0.1)
        _initialize_weights(self.conv3d_13, 0.1)
        _initialize_weights(self.bn3d_14, 0.1)
        _initialize_weights(self.conv3d_14, 0.1)
        _initialize_weights(self.bn3d_15, 0.1)
        _initialize_weights(self.conv3d_15, 0.1)
        _initialize_weights(self.bn3d_16, 0.1)
        _initialize_weights(self.conv3d_16, 0.1)
        _initialize_weights(self.bn3d_17, 0.1)
        _initialize_weights(self.conv3d_17, 0.1)
        _initialize_weights(self.bn3d_18, 0.1)
        _initialize_weights(self.conv3d_18, 0.1)

    def forward(self, x):
        '''x: [B, C, T, H, W]
        C: 1) 64 -> 160; 2) 160 -> 256
        T: 1) 7 -> 7; 2) 7 -> 7 - 2 * 3 = 1 (t_reduce=True)'''
        x1 = self.conv3d_1(F.relu(self.bn3d_1(x), inplace=True))
        x1 = self.conv3d_2(F.relu(self.bn3d_2(x1), inplace=True))
        x1 = torch.cat((x, x1), 1)

        x2 = self.conv3d_3(F.relu(self.bn3d_3(x1), inplace=True))
        x2 = self.conv3d_4(F.relu(self.bn3d_4(x2), inplace=True))
        x2 = torch.cat((x1, x2), 1)

        x3 = self.conv3d_5(F.relu(self.bn3d_5(x2), inplace=True))
        x3 = self.conv3d_6(F.relu(self.bn3d_6(x3), inplace=True))
        x3 = torch.cat((x2, x3), 1)

        x4 = self.conv3d_7(F.relu(self.bn3d_7(x3), inplace=True))
        x4 = self.conv3d_8(F.relu(self.bn3d_8(x4), inplace=True))
        x4 = torch.cat((x3, x4), 1)

        x5 = self.conv3d_9(F.relu(self.bn3d_9(x4), inplace=True))
        x5 = self.conv3d_10(F.relu(self.bn3d_10(x5), inplace=True))
        x5 = torch.cat((x4, x5), 1)

        x6 = self.conv3d_11(F.relu(self.bn3d_11(x5), inplace=True))
        x6 = self.conv3d_12(F.relu(self.bn3d_12(x6), inplace=True))
        x6 = torch.cat((x5, x6), 1)

        x7 = self.conv3d_13(F.relu(self.bn3d_13(x6), inplace=True))
        x7 = self.conv3d_14(F.relu(self.bn3d_14(x7), inplace=True))
        x7 = torch.cat((x6, x7), 1)

        x8 = self.conv3d_15(F.relu(self.bn3d_15(x7), inplace=True))
        x8 = self.conv3d_16(F.relu(self.bn3d_16(x8), inplace=True))
        x8 = torch.cat((x7, x8), 1)

        x9 = self.conv3d_17(F.relu(self.bn3d_17(x8), inplace=True))
        x9 = self.conv3d_18(F.relu(self.bn3d_18(x9), inplace=True))
        x9 = torch.cat((x8, x9), 1)
        return x9


class DUF_28L(nn.Module):
    ''' same structure as that in the paper
    https://github.com/yhjo09/VSR-DUF'''

    def __init__(self, adapt_official=False):
        super(DUF_28L, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dense_block_1 = DenseBlock_28L_A(64, 16)  # 64 + 16 * 9 = 208, T = 7
        self.dense_block_2 = DenseBlock(208, 16, t_reduce=True)  # 160 + 16 * 3 = 256, T = 1
        self.bn3d_2 = nn.BatchNorm3d(256, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(256,
                                  256, (1, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(0, 1, 1),
                                  bias=True)

        self.conv3d_r1 = nn.Conv3d(256,
                                   256, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_r2 = nn.Conv3d(256,
                                   3 * 16, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)

        self.conv3d_f1 = nn.Conv3d(256,
                                   512, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_f2 = nn.Conv3d(512,
                                   1 * 5 * 5 * 16, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)

        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 5, 5))

        # initialization
        _initialize_weights(self.conv3d_1, 0.1)
        _initialize_weights(self.bn3d_2, 0.1)
        _initialize_weights(self.conv3d_2, 0.1)
        _initialize_weights(self.conv3d_r1, 0.1)
        _initialize_weights(self.conv3d_r2, 0.1)
        _initialize_weights(self.conv3d_f1, 0.1)
        _initialize_weights(self.conv3d_f2, 0.1)

        self.adapt_official = adapt_official

    def forward(self, x):
        '''
        x: [B, T, C, H, W], T = 7. reshape to [B, C, T, H, W] for Conv3D
        Generate filters and image residual:
        Fx: [B, 25, 16, H, W] for DynamicUpsamplingFilter_3C
        Rx: [B, 3*16, 1, H, W]
        '''
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] for Conv3D
        x_center = x[:, :, T // 2, :, :]
        x = self.conv3d_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)  # reduce T to 1
        x = F.relu(self.conv3d_2(F.relu(self.bn3d_2(x), inplace=True)), inplace=True)

        # image residual
        Rx = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))  # [B, 3*16, 1, H, W]
        # filter
        Fx = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))  #[B, 25*16, 1, H, W]
        Fx = F.softmax(Fx.view(B, 25, 16, H, W), dim=1)

        # Adapt to official model weights
        if self.adapt_official:
            x = Rx.clone()
            x1 = x[:, ::3, :, :]
            x2 = x[:, 1::3, :, :]
            x3 = x[:, 2::3, :, :]

            Rx[:, :16, :, :] = x1
            Rx[:, 16:32, :, :] = x2
            Rx[:, 32:, :, :] = x3

        # dynamic filter
        out = self.dynamic_filter(x_center, Fx)  # [B, 3*R, H, W]
        out += Rx.squeeze_(2)
        out = F.pixel_shuffle(out, 4)  # [B, 3, H, W]
        return out


class DenseBlock_52L_A(nn.Module):
    '''Dense block w/ BN
    for the second denseblock, t_reduced = True'''

    def __init__(self, nf=64, ng=16):
        super(DenseBlock_52L_A, self).__init__()
        pad = (1, 1, 1)

        dense_block_l = []
        for i in range(0, 21):
            dense_block_l.append(nn.BatchNorm3d(nf + i * ng, eps=1e-3, momentum=1e-3))
            dense_block_l.append(nn.ReLU())
            dense_block_l.append(
                nn.Conv3d(nf + i * ng,
                          nf + i * ng, (1, 1, 1),
                          stride=(1, 1, 1),
                          padding=(0, 0, 0),
                          bias=True))

            dense_block_l.append(nn.BatchNorm3d(nf + i * ng, eps=1e-3, momentum=1e-3))
            dense_block_l.append(nn.ReLU())
            dense_block_l.append(
                nn.Conv3d(nf + i * ng, ng, (3, 3, 3), stride=(1, 1, 1), padding=pad, bias=True))

        # initialization
        for m in dense_block_l:
            _initialize_weights(m, 0.1)

        self.dense_blocks = nn.ModuleList(dense_block_l)

    def forward(self, x):
        for i in range(0, len(self.dense_blocks), 6):
            y = x.clone()
            for j in range(6):
                y = self.dense_blocks[i + j](y)
            x = torch.cat((x, y), 1)
        return x


class DUF_52L(nn.Module):
    ''' same structure as that in the paper
    https://github.com/yhjo09/VSR-DUF'''

    def __init__(self, adapt_official=False):
        super(DUF_52L, self).__init__()
        self.conv3d_1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dense_block_1 = DenseBlock_52L_A(64, 16)  # 64 + 21 * 9 = 400, T = 7
        self.dense_block_2 = DenseBlock(400, 16, t_reduce=True)  # 400 + 16 * 3 = 448, T = 1

        self.bn3d_2 = nn.BatchNorm3d(448, eps=1e-3, momentum=1e-3)
        self.conv3d_2 = nn.Conv3d(448,
                                  256, (1, 3, 3),
                                  stride=(1, 1, 1),
                                  padding=(0, 1, 1),
                                  bias=True)

        self.conv3d_r1 = nn.Conv3d(256,
                                   256, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_r2 = nn.Conv3d(256,
                                   3 * 16, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)

        self.conv3d_f1 = nn.Conv3d(256,
                                   512, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)
        self.conv3d_f2 = nn.Conv3d(512,
                                   1 * 5 * 5 * 16, (1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0),
                                   bias=True)

        self.dynamic_filter = DynamicUpsamplingFilter_3C((1, 5, 5))

        # initialization
        _initialize_weights(self.conv3d_1, 0.1)
        _initialize_weights(self.bn3d_2, 0.1)
        _initialize_weights(self.conv3d_2, 0.1)
        _initialize_weights(self.conv3d_r1, 0.1)
        _initialize_weights(self.conv3d_r2, 0.1)
        _initialize_weights(self.conv3d_f1, 0.1)
        _initialize_weights(self.conv3d_f2, 0.1)

        self.adapt_official = adapt_official

    def forward(self, x):
        '''
        x: [B, T, C, H, W], T = 7. reshape to [B, C, T, H, W] for Conv3D
        Generate filters and image residual:
        Fx: [B, 25, 16, H, W] for DynamicUpsamplingFilter_3C
        Rx: [B, 3*16, 1, H, W]
        '''
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W] for Conv3D
        x_center = x[:, :, T // 2, :, :]
        x = self.conv3d_1(x)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        x = F.relu(self.conv3d_2(F.relu(self.bn3d_2(x), inplace=True)), inplace=True)

        # image residual
        Rx = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))  # [B, 3*16, 1, H, W]

        # filter
        Fx = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))  #[B, 25*16, 1, H, W]
        Fx = F.softmax(Fx.view(B, 25, 16, H, W), dim=1)

        # Adapt to official model weights
        if self.adapt_official:
            x = Rx.clone()
            x1 = x[:, ::3, :, :]
            x2 = x[:, 1::3, :, :]
            x3 = x[:, 2::3, :, :]

            Rx[:, :16, :, :] = x1
            Rx[:, 16:32, :, :] = x2
            Rx[:, 32:, :, :] = x3

        # dynamic filter
        out = self.dynamic_filter(x_center, Fx)  # [B, 3*R, H, W]
        out += Rx.squeeze_(2)
        out = F.pixel_shuffle(out, 4)  # [B, 3, H, W]
        return out
