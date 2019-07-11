import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .VideoSR_base_model import VideoSRBaseModel
from models.modules.loss import CharbonnierLoss

logger = logging.getLogger('base')


class VSRBaseModel(VideoSRBaseModel):
    def __init__(self, opt):
        super(VSRBaseModel, self).__init__(opt)

    def feed_data(self, data, need_GT=True, need_BIX2=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)
        if need_BIX2:
            self.real_BIX2 = data['BIX2s'].to(self.device)

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        if isinstance(self.fake_H, tuple):
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H[0], self.real_BIX2)
            l_pix += self.l_pix_w * self.cri_pix(self.fake_H[1], self.real_H)
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def get_current_visuals(self, need_GT=True, need_BIX2=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['restore'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['BIX2'] = self.real_BIX2.detach()[0].float().cpu()
        return out_dict

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
