'''PyTorch implementation of TOFlow
Paper: Xue et al., Video Enhancement with Task-Oriented Flow, 2017
Code reference:
 1. https://github.com/anchen1011/toflow
 2. https://github.com/Coldog2333/pytoflow
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).type_as(x)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).type_as(x)
    return (x - mean) / std


def denormalize(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).type_as(x)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).type_as(x)
    return x * std + mean


def backward_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, 2, H, W), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    flow = flow.permute(0, 2, 3, 1)  # [N, 2, H, W] -> [N, H, W, 2]
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()

    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float().type_as(x)  # W(x), H(y), 2

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    output = F.grid_sample(x, vgrid, mode=interp_mode, padding_mode=padding_mode)
    return output


class SpyNet_Block(nn.Module):
    '''A submodule of SpyNet. In this implementation, 4 such submodules are used'''

    def __init__(self):
        super(SpyNet_Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, x):
        '''
        input: x: [ref im, supp im, initial flow] - (B, 8, H, W)
        output: estimated flow - (B, 2, H, W)
        '''
        return self.block(x)


class SpyNet(nn.Module):
    '''SpyNet for estimating optical flow
    Ranjan et al., Optical Flow Estimation using a Spatial Pyramid Network, 2016'''

    def __init__(self):
        super(SpyNet, self).__init__()

        self.blocks = nn.ModuleList([SpyNet_Block() for _ in range(4)])

    def forward(self, ref, supp):
        '''Estimating optical flow in coarse level, upsample, and estimate in fine level
        input: ref: reference image - [B, 3, H, W]
               supp: the image to be warped - [B, 3, H, W]
        output: estimated optical flow - [B, 2, H, W]
        '''
        B, C, H, W = ref.size()
        ref = [ref]
        supp = [supp]

        for _ in range(3):
            ref.insert(
                0,
                nn.functional.avg_pool2d(input=ref[0], kernel_size=2, stride=2,
                                         count_include_pad=False))
            supp.insert(
                0,
                nn.functional.avg_pool2d(input=supp[0], kernel_size=2, stride=2,
                                         count_include_pad=False))

        flow = torch.zeros(B, 2, H // 16, W // 16).type_as(ref[0])

        for i in range(4):
            flow_up = nn.functional.interpolate(input=flow, scale_factor=2, mode='bilinear',
                                                align_corners=True) * 2.0

            # if the sizes of upsampling and downsampling are not the same, apply zero-padding.
            if flow_up.size(2) != ref[i].size(2):
                flow_up = nn.functional.pad(input=flow_up, pad=[0, 0, 0, 1], mode='replicate')
            if flow_up.size(3) != ref[i].size(3):
                flow_up = nn.functional.pad(input=flow_up, pad=[0, 1, 0, 0], mode='replicate')

            flow = flow_up + self.blocks[i](torch.cat(
                [ref[i], backward_warp(supp[i], flow_up), flow_up], 1))
        return flow


class TOFlow(nn.Module):
    def __init__(self, adapt_official=False):
        super(TOFlow, self).__init__()

        self.SpyNet = SpyNet()

        self.conv_3x7_64_9x9 = nn.Conv2d(3 * 7, 64, 9, 1, 4)
        self.conv_64_64_9x9 = nn.Conv2d(64, 64, 9, 1, 4)
        self.conv_64_64_1x1 = nn.Conv2d(64, 64, 1)
        self.conv_64_3_1x1 = nn.Conv2d(64, 3, 1)

        self.relu = nn.ReLU(inplace=True)

        self.adapt_official = adapt_official  # True if using translated official weights else False

    def forward(self, x):
        """
        input: x: input frames - [B, 7, 3, H, W]
        output: SR reference frame - [B, 3, H, W]
        """

        B, T, C, H, W = x.size()
        x = normalize(x.view(-1, C, H, W)).view(B, T, C, H, W)

        x_ref = x[:, 3, :, :, :]
        ref_idx = 3

        # In the official torch code, the 0-th frame is the reference frame
        if self.adapt_official:
            x = x[:, [3, 0, 1, 2, 4, 5, 6], :, :, :]
            ref_idx = 0

        x_warped = []
        for i in range(7):
            if i == ref_idx:
                x_warped.append(x_ref)
            else:
                x_supp = x[:, i, :, :, :]
                flow = self.SpyNet(x_ref, x_supp)
                x_warped.append(backward_warp(x_supp, flow))
        x_warped = torch.stack(x_warped, dim=1)

        x = x_warped.view(B, -1, H, W)
        x = self.relu(self.conv_3x7_64_9x9(x))
        x = self.relu(self.conv_64_64_9x9(x))
        x = self.relu(self.conv_64_64_1x1(x))
        x = self.conv_64_3_1x1(x) + x_ref

        return denormalize(x)