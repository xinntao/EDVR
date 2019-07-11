import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from models.modules.EDVR_arch import PCD_Align, TSA_Fusion
try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class DownBlock(nn.Module):
    def __init__(self, in_channels=3, nf=32, out_channels=3,
                 negval=0.1, act=None):
        super(DownBlock, self).__init__()

        if act is None:
            act = nn.LeakyReLU(negative_slope=negval, inplace=True)
        
        res_block = mutil.ResidualBlock_noBN(nf)
        conv = nn.Conv2d(nf, out_channels, kernel_size=3, 
                         stride=2, padding=1, bias=True)

        self.down_block = nn.Sequential(res_block, conv)

    def forward(self, x):
        x = self.down_block(x)
        return x

class ForwardConcat(nn.Module):
    """ This module is used to compute the forward procedure 
        of 4 modules, and then concat the output together. """
    def __init__(self, module, module_nums=4, *args):
        super(ForwardConcat, self).__init__()
        self.modules_num = 4
        self.modules_list = nn.ModuleList([
            module(*args) for _ in range(module_nums)
        ])
    
    def forward(self, x):
        for i in range(self.modules_num):
            if i == 0: 
                output = self.modules_list[i](x)
            else:
                output = torch.cat((self.modules_list[i](x), output), dim=1)
        return output

class PCD_Align_With_Offset(PCD_Align):
    """Use to align the feature only in one scale"""
    def __init__(self, nf=32, groups=8):
        super(PCD_Align_With_Offset, self).__init__(nf, groups)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in original spatial size
        nbr_fea, ref_fea, each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bicubic', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bicubic', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bicubic', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bicubic', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        return L1_fea, L1_offset


class TSA_align_Bicubic(TSA_Fusion):
    def __index__(self, nf=64, nframes=5, center=2):
        super(TSA_align_Bicubic, self).__init__(nf, nframes, center)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bicubic', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bicubic', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea

class DCN_Align(nn.Module):
    """Use to align the feature only in one scale"""
    def __init__(self, nf=64, groups=8):
        super(DCN_Align, self).__init__()
        # original spatial size
        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # low sptial size
        self.low_offset_conv = nn.Conv2d(2*nf, nf, 1, 1, bias=True)
        self.low_fea_conv = nn.Conv2d(2*nf, nf, 1, 1, bias=True)

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.fea_conv = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                   deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea, ref_fea, low_fea, low_offset):
        '''align other neighboring frames to the reference frame in original spatial size
        nbr_fea with [B,2*C,H,W] features and ref_fea with [B,C,H,W] features
        '''
        offset = torch.cat([nbr_fea, ref_fea], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        low_offset = self.low_offset_conv(low_offset)
        low_offset = F.interpolate(low_offset, scale_factor=2, mode='bicubic', align_corners=False)
        
        offset = self.lrelu(self.offset_conv2(torch.cat([offset, low_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        aligned_fea = self.dcnpack(nbr_fea, offset)
        
        low_fea = self.low_fea_conv(low_fea)
        low_fea = F.interpolate(low_fea, scale_factor=2, mode='bicubic', align_corners=False)
        aligned_fea = self.lrelu(self.fea_conv(torch.cat([aligned_fea, low_fea], dim=1)))

        # Cascading 
        offset = torch.cat([aligned_fea, ref_fea], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        aligned_fea = self.lrelu(self.cas_dcnpack(aligned_fea, offset))
        
        return aligned_fea, offset


class VSRUN(nn.Module):
    def __init__(self, nf=32, nframes=5, groups=8, res_blocks=10, 
                 res_groups=2, center=None, w_TSA=True):
        super(VSRUN, self).__init__()
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

        # align module
        self.pcd_align = PCD_Align_With_Offset(nf=4*nf, groups=groups)
        self.dcn_align1 = DCN_Align(nf=2*nf, groups=groups)
        self.dcn_align2 = DCN_Align(nf=nf, groups=groups)

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
    
    def align_pcd(self, L1_fea, L2_fea, L3_fea):
        """align the video frames feature using PCD algin module"""
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea, offset  = [], []
        for i in range(self.nframes):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            results = self.pcd_align(nbr_fea_l, ref_fea_l)
            aligned_fea.append(results[0])
            offset.append(results[1])
        aligned_fea = torch.stack(aligned_fea, dim=1)
        offset = torch.stack(offset, dim=1)
        return aligned_fea, offset
    
    def align_dcn(self, align_module, feature, lower_fea, lower_offset):
        """align the video frames feature using DCN align module"""
        ref_fea = feature[:, self.center, :, :, :].clone()
        aligned_fea, offset  = [], []
        for i in range(self.nframes):
            nbr_fea_i =  feature[:, i, :, :, :].clone()
            lower_fea_i = lower_fea[:, i, :, :, :].clone()
            lower_offset_i = lower_offset[:, i, :, :, :].clone()
            results = align_module(nbr_fea_i, ref_fea, lower_fea_i, lower_offset_i)
            aligned_fea.append(results[0])
            offset.append(results[1])
        aligned_fea = torch.stack(aligned_fea, dim=1)
        offset = torch.stack(offset, dim=1)
        return aligned_fea, offset
    
    def fusion(self, fusion_module, aligned_fea, size):
        """fuse the aligned video frames freatures"""
        B, _, H, W = size
        if not self.w_TSA:
            return fusion_module(aligned_fea.view(B, -1, H, W))
        return fusion_module(aligned_fea)
        
    def forward(self, inputs):
        B, N, C, H, W = inputs.size() # N: n_frames
        x = self.upsample(inputs.view(-1, C, H, W))
        x_head = self.head(x)

        # down sample
        x_down1 = self.down1(x_head)
        L1_fea = self.down2(x_down1)
        L2_fea = self.pcd_down1(L1_fea)
        L3_fea = self.pcd_down2(L2_fea)

        # resize tensors' shape
        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)
        
        # Frames PCD Align and Fusion 1
        aligned_fea, offset = self.align_pcd(L1_fea, L2_fea, L3_fea)
        x_fusion = self.fusion(self.tsa_fusion1, aligned_fea, (B, -1, H, W))
        
        # Reconstruction and upscale 2x
        x_up1 = self.up1(x_fusion + self.recon_trunk1(x_fusion))

        # Frames DCN Align and Fusion 2
        x_down1 = x_down1.view(B, N, -1, 2 * H, 2 * W)
        aligned_fea, offset = self.align_dcn(self.dcn_align1, x_down1, aligned_fea, offset)
        x_fusion = self.fusion(self.tsa_fusion2, aligned_fea, (B, -1, 2 * H, 2 * W))

        # concat features and upscale (2x size of inputs)
        x_cat_2x = torch.cat((x_fusion, x_up1), -3)
        x_up2 = self.up2(x_cat_2x + self.recon_trunk2(x_cat_2x))

        # frames DCN Align and fusion 3
        ref_fea = x_head.view(B, N, -1, 4 * H, 4 * W)
        aligned_fea, offset = self.align_dcn(self.dcn_align2, ref_fea, aligned_fea, offset)
        x_fusion = self.fusion(self.tsa_fusion3, aligned_fea, (B, -1, 4 * H, 4 * W))

        # concat x_fusion and x_up2 (4x size of inputs)
        x_cat_4x = torch.cat((x_fusion, x_up2), -3)

        # output
        out_2x = self.tail1(x_cat_2x)
        out_4x = self.tail2(x_cat_4x)

        return out_2x, out_4x