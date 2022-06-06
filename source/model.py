import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import select_resnet


class Model(nn.Module):

    def __init__(self, num_seq=8, seq_len=5, pred_step=1, network='resnet50'):
        super(Model, self).__init__()
        torch.cuda.manual_seed(233)

        # self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        # self.last_duration = int(math.ceil(seq_len / 4))
        # self.last_size = int(math.ceil(sample_size / 32))
        # print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network)
        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.agg = nn.GRU(input_size=self.param['feature_size'],
                          hidden_size=self.param['hidden_size'],
                          num_layers=self.param['num_layers'],
                          batch_first=True)

        self.network_pred = nn.Sequential(
            nn.Linear(self.param['feature_size'], self.param['feature_size']),
            nn.BatchNorm1d(self.param['feature_size']),
            nn.ReLU(inplace=True),
            nn.Linear(self.param['feature_size'], self.param['feature_size'])
        )

        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(self.param['num_layers'], batch_size, self.param['feature_size'], device=device)
        return h0

    def reset_mask(self):
        self.mask = None

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        # block = block.view(B * N, C, SL, H, W)
        block = block.permute(0, 1, 3, 2, 4, 5).contiguous().view(B * N * SL, C, H, W)
        feature = self.backbone(block) # [B*N*SL,2048,1,1]
        del block
        # feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        feature_inf_all = feature.view(B, N, SL, self.param['feature_size'])  # [B,N,SL,2048], before ReLU, (-inf, +inf)
        feature = self.relu(feature)  # [0, +inf)
        feature = feature.view(B * N, SL, self.param['feature_size'])  # [B*N,SL,2048], [0, +inf)
        # feature_inf = feature_inf_all[:, :, -1, :].contiguous()  # [B,N,2048]
        # del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, 0:SL-self.pred_step, :], self._get_initial_state(feature.size(0), feature.device))
        hidden = hidden[:, -1, :]  # [B*N,2048]
        pred = self.network_pred(hidden)  # [B*N,2048]
        del hidden

        ### Get similarity score ###
        # # pred: [B, pred_step, D, last_size, last_size]
        # # GT: [B, N, D, last_size, last_size]
        feature_inf = feature_inf_all.view(B * N * SL, self.param['feature_size']).transpose(0, 1)
        score = torch.matmul(pred, feature_inf).view(B, N, self.pred_step, B, N, SL)
        del feature_inf, pred

        if self.mask is None:  # only compute mask once
            # mask meaning: -1: distortion, 0: easy content, 1: pos, -2: hard content
            mask = torch.zeros((B, N, self.pred_step, B, N, SL), dtype=torch.int8, requires_grad=False).detach().cuda()
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -2  # hard content

            for k in range(B):
                mask[k, :, torch.arange(self.pred_step), k, :, torch.arange(SL - self.pred_step, N)] = -1  # distortion

            tmp = mask.view(B * N, self.pred_step, B * N, SL)
            for j in range(B * N):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(SL - self.pred_step, N)] = 1  # pos

            mask = tmp.view(B, N, self.pred_step, B, N, SL)

            self.mask = mask

        return [score, self.mask]



