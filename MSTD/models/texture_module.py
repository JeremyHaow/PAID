# models/texture_module.py
# 纹理分析模块

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TextureContrastModule(nn.Module):
    """
    提取图像中丰富和贫乏纹理区域，并计算对比度特征
    基于PatchCraft论文的方法
    """
    def __init__(self, patch_size=32, num_patches=64):
        super(TextureContrastModule, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        
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
        """提取丰富和贫乏纹理区域"""
        B, C, H, W = x.shape
        
        # 确保H和W足够大以提取patches
        assert H >= self.patch_size and W >= self.patch_size
        
        all_patches = []
        diversity_scores = []
        
        # 随机提取patches并计算其多样性得分
        for _ in range(self.num_patches * 2):  # 提取更多以便选择
            # 随机选择patch左上角位置
            h_idx = random.randint(0, H - self.patch_size)
            w_idx = random.randint(0, W - self.patch_size)
            
            # 提取patch
            patch = x[:, :, h_idx:h_idx+self.patch_size, w_idx:w_idx+self.patch_size]
            all_patches.append(patch)
            
            # 计算纹理多样性得分（基于梯度变化）
            dx = patch[:, :, :, 1:] - patch[:, :, :, :-1]
            dy = patch[:, :, 1:, :] - patch[:, :, :-1, :]
            
            gradient_magnitude = torch.sqrt(dx.pow(2).sum(dim=1) + dy.pow(2).sum(dim=1))
            diversity_score = gradient_magnitude.mean()
            diversity_scores.append(diversity_score)
        
        # 将diversity_scores转换为张量
        diversity_scores = torch.stack(diversity_scores)
        
        # 根据多样性得分排序
        _, indices = torch.sort(diversity_scores, descending=True)
        
        # 选择多样性最高的n个patch作为丰富纹理区域
        rich_indices = indices[:self.num_patches]
        rich_patches = [all_patches[i] for i in rich_indices]
        
        # 选择多样性最低的n个patch作为贫乏纹理区域
        poor_indices = indices[-self.num_patches:]
        poor_patches = [all_patches[i] for i in poor_indices]
        
        # 堆叠patch创建丰富和贫乏纹理图像
        rich_texture = torch.cat(rich_patches, dim=0)
        poor_texture = torch.cat(poor_patches, dim=0)
        
        return rich_texture, poor_texture
    
    def forward(self, rich_texture, poor_texture):
        """计算丰富和贫乏纹理区域之间的对比特征"""
        # 提取特征
        rich_features = self.feature_extractor(rich_texture)
        poor_features = self.feature_extractor(poor_texture)
        
        # 计算平均特征
        rich_features = rich_features.mean(dim=0, keepdim=True)
        poor_features = poor_features.mean(dim=0, keepdim=True)
        
        # 计算对比特征
        contrast_features = torch.cat([rich_features, poor_features], dim=1)
        contrast_features = self.contrast_extractor(contrast_features)
        
        return contrast_features