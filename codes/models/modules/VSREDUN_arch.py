import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.VSRUN_arch import DownBlock, ForwardConcat
from models.modules.EDVR_arch import PCD_Align, TSA_Fusion


class VSREDUN(nn.Module):
    def __init__(self, nf=32, nframes=5, groups=8, res_blocks=10, 
                 res_groups=2, center=None, w_TSA=True):
        super(VSREDUN, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.nframes = nframes
        self.w_TSA = w_TSA
        ReconstructionBlock1 = functools.partial(mutil.ResidualGroup, nf=4*nf,
                                                 res_blocks=res_blocks//res_groups, reduction=nf)
        ReconstructionBlock2 = functools.partial(mutil.ResidualGroup, nf=4*nf,
                                                 res_blocks=res_blocks//res_groups, reduction=nf)

        self.head = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.upsample = nn.Upsample(scale_factor=4,
                                    mode='bicubic', align_corners=False)
        
        self.down1 = DownBlock(3, nf, 2*nf)
        self.down2 = DownBlock(2*nf, 2*nf, 4*nf)
        self.pcd_down1 = DownBlock(4*nf, 4*nf, 4*nf)
        self.pcd_down2 = DownBlock(4*nf, 4*nf, 4*nf)
        
        self.pcd_L2_conv1 = nn.Conv2d(4*nf, 2*nf, 3, 1, 1, bias=True)
        self.pcd_L1_conv1 = nn.Conv2d(4*nf, 2*nf, 3, 1, 1, bias=True)
        self.pcd_L1_conv2 = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.pcd_D1_conv2 = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)

        # align module
        self.pcd_align1 = PCD_Align(nf=4*nf, groups=groups)
        self.pcd_align2 = PCD_Align(nf=2*nf, groups=groups)
        self.pcd_align3 = PCD_Align(nf=nf, groups=groups)

        if self.w_TSA:
            self.tsa_fusion1 = TSA_Fusion(nf=4*nf, nframes=nframes, center=self.center)
            self.tsa_fusion2 = TSA_Fusion(nf=2*nf, nframes=nframes, center=self.center)
            self.tsa_fusion3 = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion1 = nn.Conv2d(nframes * 4 * nf, 4 * nf, 1, 1, bias=True)
            self.tsa_fusion2 = nn.Conv2d(nframes * 2 * nf, 2 * nf, 1, 1, bias=True)
            self.tsa_fusion3 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        
        self.recon_trunk1 = mutil.make_layer(ReconstructionBlock1, res_groups//2)
        self.recon_trunk2 = mutil.make_layer(ReconstructionBlock2, res_groups//2)
        
        self.up1 = nn.Sequential(
            ForwardConcat(mutil.ResidualBlock_noBN, 4, 4*nf),
            nn.Conv2d(16*nf, 8*nf, 1, 1, 0, bias=True),
            nn.PixelShuffle(2)
        )
        self.up2 = nn.Sequential(
            ForwardConcat(mutil.ResidualBlock_noBN, 4, 4*nf),
            nn.Conv2d(16*nf, 4*nf, 1, 1, 0, bias=True),
            nn.PixelShuffle(2)
        )

        self.tail1 = nn.Conv2d(4*nf, 3, 3, 1, 1, bias=True)
        self.tail2 = nn.Conv2d(2*nf, 3, 3, 1, 1, bias=True)
    
    def align_pcd(self, align_module, L1_fea, L2_fea, L3_fea):
        """align the video frames feature using PCD algin module"""
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(self.nframes):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            results = align_module(nbr_fea_l, ref_fea_l)
            aligned_fea.append(results)
        aligned_fea = torch.stack(aligned_fea, dim=1)
        return aligned_fea
    
    def fusion(self, fusion_module, aligned_fea, size):
        """fuse the aligned video frames freatures"""
        B, _, H, W = size
        if not self.w_TSA:
            return fusion_module(aligned_fea.view(B, -1, H, W))
        return fusion_module(aligned_fea)
        
    def forward(self, inputs):
        B, N, C, H, W = inputs.size() # N: n_frames
        x = self.upsample(inputs.view(-1, C, H, W))
        
        # feature extraction
        x_head = self.head(x)

        # down sample
        x_down = self.down1(x_head)
        L1_down = self.down2(x_down)
        L2_down = self.pcd_down1(L1_down)
        L3_down = self.pcd_down2(L2_down)

        # resize tensors' shape
        L1_fea = L1_down.view(B, N, -1, H, W)
        L2_fea = L2_down.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_down.view(B, N, -1, H // 4, W // 4)
        
        # # Frames PCD Align and Fusion in lowest level
        aligned_fea = self.align_pcd(self.pcd_align1, L1_fea, L2_fea, L3_fea)
        x_fusion = self.fusion(self.tsa_fusion1, aligned_fea, (B, -1, H, W))
        
        # Reconstruct and upscale to 2x
        x_up1 = self.up1(self.recon_trunk1(x_fusion))

        # # Frames PCD Align and Fusion in 2x level
        L1_down = self.pcd_L1_conv1(L1_down)
        L2_down = self.pcd_L2_conv1(L2_down)
        
        x_down1 = x_down.view(B, N, -1, 2 * H, 2 * W)
        L1_fea = L1_down.view(B, N, -1, H, W)
        L2_fea = L2_down.view(B, N, -1, H // 2, W // 2)

        aligned_fea = self.align_pcd(self.pcd_align2, x_down1, L1_fea, L2_fea)
        x_fusion = self.fusion(self.tsa_fusion2, aligned_fea, (B, -1, 2 * H, 2 * W))

        # concatenate features and upscale to 4x
        x_cat_2x = torch.cat((x_fusion, x_up1), -3)
        x_up2 = self.up2(self.recon_trunk2(x_cat_2x))

        # frames PCD Align and fusion in 4x level
        L1_down = self.pcd_L1_conv2(L1_down)
        x_down2 = self.pcd_D1_conv2(x_down)
        
        L1_fea = L1_down.view(B, N, -1, H, W)
        x_down2 = x_down2.view(B, N, -1, 2 * H, 2 * W)
        x_head = x_head.view(B, N, -1, 4 * H, 4 * W)

        aligned_fea = self.align_pcd(self.pcd_align3, x_head, x_down2, L1_fea)
        x_fusion = self.fusion(self.tsa_fusion3, aligned_fea, (B, -1, 4 * H, 4 * W))

        # concatenate features
        x_cat_4x = torch.cat((x_fusion, x_up2), -3)

        # output
        out_2x = self.tail1(x_cat_2x)
        out_4x = self.tail2(x_cat_4x)

        return out_4x # out_2x, out_4x