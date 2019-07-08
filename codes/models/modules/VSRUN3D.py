import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.EDVR_arch import PCD_Align, TSA_Fusion
from models.modules.VSRUN import ForwardConcat

class DownBlock3D(nn.Module):
    def __init__(self, in_channels=3, nf=32, out_channels=3,
                 negval=0.2, act=None):
        super(DownBlock3D, self).__init__()

        if act is None:
            act = nn.LeakyReLU(negative_slope=negval, inplace=True)
        
        res_block = ResidualBlock3D(nf)
        conv = nn.Conv3d(nf, out_channels, kernel_size=3,
                         stride=2, padding=1, bias=True)

        self.down_block = nn.Sequential(res_block, conv)

    def forward(self, x):
        x = self.down_block(x)
        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class VSRUN3D(nn.Module):
    def __init__(self, nf=32, nframes=5, groups=8, back_RBs=10, center=None,
                 w_TSA=True):
        super(VSRUN3D, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        ResidualBlock3D_f1 = functools.partial(ResidualBlock3D, nf=4*nf)
        ResidualBlock3D_f2 = functools.partial(ResidualBlock3D, nf=2*nf)

        self.head = nn.Conv3d(3, nf, 3, 1, 1, bias=True)
        self.upsample = nn.Upsample(scale_factor=4,
                                    mode='bicubic', align_corners=False)
        self.down1 = DownBlock3D(3, nf, 2*nf)
        self.down2 = DownBlock3D(2*nf, 2*nf, 4*nf)
        
        self.res_blocks1 = mutil.make_layer(ResidualBlock3D_f1, back_RBs//2)
        self.res_blocks2 = mutil.make_layer(ResidualBlock3D_f2, back_RBs//2)
        
        self.up1 = nn.Sequential(
            ForwardConcat(ResidualBlock3D, 4, 4*nf),
            nn.Conv3d(4*nf, 2*nf, 1, 1, 1, bias=True),
            nn.PixelShuffle(2)
        )
        self.up2 = nn.Sequential(
            ForwardConcat(ResidualBlock3D, 4, 2*nf),
            nn.Conv3d(2*nf, nf, 1, 1, 1, bias=True),
            nn.PixelShuffle(2)
        )

        self.tail1 = nn.Conv2d(4 * nframes * nf, 3, 3, 1, 1, bias=True)
        self.tail2 = nn.Conv2d(2 * nframes * nf, 3, 3, 1, 1, bias=True)
    
    def forward(self, inputs):
        B, _, C, H, W = inputs.size()
        x = self.upsample(inputs.view(-1, C, H, W))
        x_head = self.head(x.view(B, -1, C, 4 * H, 4 * W))

        x_down1 = self.down1(x_head)
        x_down2 = self.down2(x_down1)

        x = self.res_blocks1(x_down2)
        x_up1 = self.up1(x)

        # concat x_down1 and x_up1 (2x size of inputs)
        x_cat_2x = torch.cat((x_down1, x_up1), -3)
        x = self.res_blocks2(x_cat_2x)
        x_up2 = self.up2(x)

        # concat x_down1 and x_up1 (4x size of inputs)
        x_cat_4x = torch.cat((x_head, x_up2), -3)

        # output
        B, _, _, H, W = x_cat_2x.size()
        out_2x = self.tail1(x_cat_2x.view(B, -1, H, W))
        out_4x = self.tail2(x_cat_4x.view(B, -1, 2 * H, 2 * W))

        return out_2x, out_4x