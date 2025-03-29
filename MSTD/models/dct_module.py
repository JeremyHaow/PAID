# models/dct_module.py
# DCT变换模块，用于频域特征提取

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def DCT_mat(size):
    """生成离散余弦变换矩阵"""
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    """生成频域过滤器"""
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    """Sigmoid标准化"""
    return 2. * torch.sigmoid(x) - 1.

class Filter(nn.Module):
    """频域滤波器"""
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class DCTTextureExtractor(nn.Module):
    """
    使用DCT变换提取图像的丰富和贫乏纹理区域
    """
    def __init__(self, window_size=32, stride=16, output=224, grade_N=6, level_filter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_filter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_filter)
        self.N = (output // window_size) * (output // window_size)
        
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)
        
        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=stride
        )
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size), 
            kernel_size=(window_size, window_size), 
            stride=window_size
        )
        
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_filter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])
        
    def extract_texture_regions(self, x):
        """
        提取图像中的丰富和贫乏纹理区域
        
        参数:
            x: 输入图像 [B, C, H, W]
            
        返回:
            rich_texture: 丰富纹理区域
            poor_texture: 贫乏纹理区域
        """
        batch_size, C, H, W = x.shape
        rich_textures = []
        poor_textures = []
        
        for b in range(batch_size):
            x_b = x[b]  # [C, H, W]
            x_minmin, x_maxmax, _, _ = self.forward(x_b)  # 提取最不丰富和最丰富的纹理区域
            
            # 调整为4D张量 [1, C, H, W]
            poor_texture = x_minmin.unsqueeze(0)
            rich_texture = x_maxmax.unsqueeze(0)
            
            rich_textures.append(rich_texture)
            poor_textures.append(poor_texture)
        
        # 连接批次维度上的结果
        rich_textures = torch.cat(rich_textures, dim=0)
        poor_textures = torch.cat(poor_textures, dim=0)
        
        return rich_textures, poor_textures
        
    def forward(self, x):
        """
        提取给定图像中的不同纹理区域
        
        参数:
            x: 输入图像 [C, H, W]
            
        返回:
            x_minmin: 最不丰富纹理区域
            x_maxmax: 最丰富纹理区域
            x_minmin1: 第二不丰富纹理区域
            x_maxmax1: 第二丰富纹理区域
        """
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, H, W = x.shape
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)  
        
        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size) 
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T
        
        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)
        
        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[1,2,3])
            grade += w * _x            
            w *= k
        
        _, idx = torch.sort(grade)
        max_idx = torch.flip(idx, dims=[0])[:N]
        maxmax_idx = max_idx[0]
        if len(max_idx) == 1:
            maxmax_idx1 = max_idx[0]
        else:
            maxmax_idx1 = max_idx[1]

        min_idx = idx[:N]
        minmin_idx = idx[0]
        if len(min_idx) == 1:
            minmin_idx1 = idx[0]
        else:
            minmin_idx1 = idx[1]

        x_minmin = torch.index_select(level_x_unfold, 0, minmin_idx)
        x_maxmax = torch.index_select(level_x_unfold, 0, maxmax_idx)
        x_minmin1 = torch.index_select(level_x_unfold, 0, minmin_idx1)
        x_maxmax1 = torch.index_select(level_x_unfold, 0, maxmax_idx1)

        x_minmin = x_minmin.reshape(1, level_N*C*window_size*window_size).transpose(0, 1)
        x_maxmax = x_maxmax.reshape(1, level_N*C*window_size*window_size).transpose(0, 1)
        x_minmin1 = x_minmin1.reshape(1, level_N*C*window_size*window_size).transpose(0, 1)
        x_maxmax1 = x_maxmax1.reshape(1, level_N*C*window_size*window_size).transpose(0, 1)

        x_minmin = self.fold0(x_minmin)
        x_maxmax = self.fold0(x_maxmax)
        x_minmin1 = self.fold0(x_minmin1)
        x_maxmax1 = self.fold0(x_maxmax1)
       
        return x_minmin, x_maxmax, x_minmin1, x_maxmax1
