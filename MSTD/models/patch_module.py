# models/patch_module.py
# Patch操作相关模块

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class PatchShuffleModule(nn.Module):
    """
    Patch Shuffle模块
    将图像分成相同大小的patch并随机打乱
    """
    def __init__(self):
        super(PatchShuffleModule, self).__init__()
    
    def forward(self, x, patch_size):
        """
        将图像分成相同大小的patch并随机打乱
        参数:
            x: 输入图像 [B, C, H, W]
            patch_size: patch的大小
        返回:
            打乱后的图像 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 确保H和W可以被patch_size整除
        H_pad = (patch_size - H % patch_size) % patch_size
        W_pad = (patch_size - W % patch_size) % patch_size
        
        if H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad))
            H, W = H + H_pad, W + W_pad
        
        # 计算每个维度的patch数
        n_h = H // patch_size
        n_w = W // patch_size
        n_patches = n_h * n_w
        
        # 将图像重塑为patches
        patches = x.view(B, C, n_h, patch_size, n_w, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, n_patches, C, patch_size, patch_size)
        
        # 为每个batch随机打乱patches
        shuffled_patches = patches.clone()
        for b in range(B):
            idx = torch.randperm(n_patches, device=x.device)
            shuffled_patches[b] = patches[b, idx]
        
        # 将打乱的patches重组为图像
        shuffled_patches = shuffled_patches.view(B, n_h, n_w, C, patch_size, patch_size)
        shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        shuffled_x = shuffled_patches.view(B, C, H, W)
        
        return shuffled_x


class AdaptivePatchSelector(nn.Module):
    """
    自适应选择最优patch大小
    这是MSTD的一个创新点，不同于SFLD使用固定patch大小
    """
    def __init__(self, candidate_sizes=[28, 56, 112]):
        super(AdaptivePatchSelector, self).__init__()
        self.candidate_sizes = candidate_sizes
        
        # 根据图像内容预测最优patch大小的网络
        self.patch_predictor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, len(candidate_sizes)),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        """
        预测每个候选patch尺寸的权重
        返回选定的patch尺寸列表（可以是多个）
        """
        weights = self.patch_predictor(x)
        
        # 选择权重最高的前两个尺寸
        _, indices = torch.topk(weights, 2, dim=1)
        
        # 获取每个batch最适合的patch大小
        selected_sizes = []
        for i in range(indices.size(0)):
            batch_sizes = [self.candidate_sizes[idx] for idx in indices[i]]
            selected_sizes.append(batch_sizes)
            
        # 汇总所有batch的结果，得到唯一的patch尺寸集合
        final_sizes = list(set([size for batch_sizes in selected_sizes for size in batch_sizes]))
        return final_sizes