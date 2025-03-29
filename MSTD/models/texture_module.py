# models/texture_module.py
# 纹理分析模块

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class Filter(nn.Module):
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

class DCT_base_Rec_Module(nn.Module):
    """
    使用DCT变换提取图像的纹理特征
    Args:
        x: [C, H, W] -> [C*level, output, output]
    """
    def __init__(self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]):
        super().__init__()
        
        assert output % window_size == 0
        assert len(level_fliter) > 0
        
        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
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
        
        lm, mh = 2.82, 2
        level_f = [
            Filter(window_size, 0, window_size * 2)
        ]
        
        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList([Filter(window_size, window_size * 2. / grade_N * i, window_size * 2. / grade_N * (i+1), norm=True) for i in range(grade_N)])
        
    def forward(self, x):
        
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, W, H = x.shape
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

        x_minmin = x_minmin.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_maxmax = x_maxmax.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_minmin1 = x_minmin1.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)
        x_maxmax1 = x_maxmax1.reshape(1, level_N*C*window_size* window_size).transpose(0, 1)

        x_minmin = self.fold0(x_minmin)
        x_maxmax = self.fold0(x_maxmax)
        x_minmin1 = self.fold0(x_minmin1)
        x_maxmax1 = self.fold0(x_maxmax1)

        return x_minmin, x_maxmax, x_minmin1, x_maxmax1

class TextureContrastModule(nn.Module):
    """
    提取图像中丰富和贫乏纹理区域，并计算对比度特征
    使用DCT变换来提取纹理特征，更有效区分真实和AI生成图像
    """
    def __init__(self, patch_size=32, num_patches=64, window_size=32, dct_output_size=224):
        super(TextureContrastModule, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # DCT模块用于提取纹理
        self.dct_module = DCT_base_Rec_Module(
            window_size=window_size,
            stride=window_size//2,
            output=dct_output_size,
            grade_N=6,
            level_fliter=[0]
        )
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 对比特征提取器
        self.contrast_extractor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def extract_texture_regions(self, x):
        """使用DCT变换提取丰富和贫乏纹理区域"""
        B, C, H, W = x.shape
        
        # 处理批次中的每张图像
        poor_textures = []
        rich_textures = []
        
        for i in range(B):
            # 对单张图像应用DCT分析
            img = x[i]  # [C, H, W]
            
            # 提取低频和高频纹理区域
            poor_texture, rich_texture, poor_texture1, rich_texture1 = self.dct_module(img)
            
            # 将结果堆叠起来
            poor_texture = torch.cat([poor_texture, poor_texture1], dim=0)
            rich_texture = torch.cat([rich_texture, rich_texture1], dim=0)
            
            # 调整维度并处理到合适的尺寸
            if poor_texture.size(0) < 3:
                poor_texture = poor_texture.repeat(3, 1, 1)
            if rich_texture.size(0) < 3:
                rich_texture = rich_texture.repeat(3, 1, 1)
            
            poor_texture = F.interpolate(
                poor_texture.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            
            rich_texture = F.interpolate(
                rich_texture.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 添加到列表
            poor_textures.append(poor_texture)
            rich_textures.append(rich_texture)
        
        # 合并批次
        poor_texture_batch = torch.cat(poor_textures, dim=0)
        rich_texture_batch = torch.cat(rich_textures, dim=0)
        
        return poor_texture_batch, rich_texture_batch
    
    def forward(self, poor_texture, rich_texture):
        """计算贫乏和丰富纹理区域之间的对比特征"""
        # 提取特征
        poor_features = self.feature_extractor(poor_texture)
        rich_features = self.feature_extractor(rich_texture)
        
        # 计算平均特征
        poor_features = poor_features.mean(dim=0, keepdim=True)
        rich_features = rich_features.mean(dim=0, keepdim=True)
        
        # 计算对比特征
        contrast_features = torch.cat([rich_features, poor_features], dim=1)
        contrast_features = self.contrast_extractor(contrast_features)
        
        return contrast_features