# 纹理分析与处理模块

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from .texture_module import DCT_base_Rec_Module
from .patch_module import PatchShuffleModule

class SRMConv(nn.Module):
    """
    SRM卷积层，用于提取图像的噪声残差特征
    """
    def __init__(self):
        super(SRMConv, self).__init__()
        
        # 初始化SRM卷积核
        srm_kernels = self._get_srm_kernels()
        self.weight = nn.Parameter(srm_kernels, requires_grad=False)
        self.register_buffer('window', torch.ones(3, 1, 5, 5))
    
    def _get_srm_kernels(self):
        # 基础的高通滤波器
        filter1 = torch.tensor([[-1, 2, -1], 
                               [2, -4, 2], 
                               [-1, 2, -1]], dtype=torch.float32)
        filter2 = torch.tensor([[0, 0, 0], 
                               [1, -2, 1], 
                               [0, 0, 0]], dtype=torch.float32)
        filter3 = torch.tensor([[0, 1, 0], 
                               [0, -2, 0], 
                               [0, 1, 0]], dtype=torch.float32)
        
        # SRM卷积核
        srm_kernel = torch.zeros((3, 3, 3, 3))
        for i in range(3):
            srm_kernel[i, i, :, :] = filter1
            srm_kernel[i, i, :, :] += filter2
            srm_kernel[i, i, :, :] += filter3
        
        return srm_kernel

    def forward(self, x):
        # 应用SRM卷积提取噪声残差特征
        out = F.conv2d(x, self.weight, padding=1)
        return out

class TextureAnalysisModule(nn.Module):
    """
    纹理分析模块：选择纹理丰富和纹理贫乏的区域，重组并通过SRM和ResNet50处理
    """
    def __init__(self, patch_size=32, output_size=224, num_patches=9, dct_window_size=32):
        super(TextureAnalysisModule, self).__init__()
        self.patch_size = patch_size
        self.output_size = output_size
        self.num_patches = num_patches
        
        # DCT模块用于纹理分析
        self.dct_module = DCT_base_Rec_Module(
            window_size=dct_window_size,
            stride=dct_window_size//2,
            output=output_size,
            grade_N=6,
            level_fliter=[0]
        )
        
        # Patch打乱模块
        self.patch_shuffle = PatchShuffleModule()
        
        # SRM卷积层
        self.srm_rich = SRMConv()
        self.srm_poor = SRMConv()
        
        # 加载预训练的ResNet50
        resnet = models.resnet50(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-1])  # 移除最后的全连接层
        
        # 新的特征嵌入层
        self.rich_embed = nn.Linear(2048, 512)
        self.poor_embed = nn.Linear(2048, 512)
        
    def extract_and_sort_patches(self, x):
        """提取并根据纹理丰富程度排序图像块"""
        B, C, H, W = x.shape
        
        # 确保图像尺寸可以被patch_size整除
        H_pad = (self.patch_size - H % self.patch_size) % self.patch_size
        W_pad = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if H_pad > 0 or W_pad > 0:
            x = F.pad(x, (0, W_pad, 0, H_pad))
            H, W = H + H_pad, W + W_pad
        
        # 计算每个维度上的patch数
        n_h = H // self.patch_size
        n_w = W // self.patch_size
        total_patches = n_h * n_w
        
        patches_scores = []
        patches = []
        
        # 将图像分割为多个patch
        for i in range(n_h):
            for j in range(n_w):
                h_start = i * self.patch_size
                w_start = j * self.patch_size
                patch = x[:, :, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
                
                # 计算纹理丰富度分数（使用梯度方差）
                dx = torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1])
                dy = torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :])
                
                grad_mean = (dx.mean() + dy.mean()) / 2.0
                grad_var = (dx.var() + dy.var()) / 2.0
                texture_score = grad_mean * grad_var
                
                patches_scores.append(texture_score.item())
                patches.append(patch)
        
        # 排序
        sorted_indices = np.argsort(patches_scores)
        
        # 选择最丰富和最不丰富的patches
        rich_indices = sorted_indices[-self.num_patches:]
        poor_indices = sorted_indices[:self.num_patches]
        
        rich_patches = [patches[i] for i in rich_indices]
        poor_patches = [patches[i] for i in poor_indices]
        
        return rich_patches, poor_patches
    
    def reconstruct_image(self, patches, rows=3, cols=3):
        """从多个patch重建图像"""
        B, C, H, W = patches[0].shape
        
        # 创建输出图像
        output = torch.zeros(B, C, rows*H, cols*W).to(patches[0].device)
        
        # 填充图像
        for idx, patch in enumerate(patches):
            if idx >= rows*cols:
                break
                
            row = idx // cols
            col = idx % cols
            
            output[:, :, row*H:(row+1)*H, col*W:(col+1)*W] = patch
        
        return output
    
    def forward(self, x):
        """前向传播"""
        B, C, H, W = x.shape
        
        # 提取丰富和贫乏纹理的patches
        rich_patches, poor_patches = self.extract_and_sort_patches(x)
        
        # 重建为两张图像
        rich_image = self.reconstruct_image(rich_patches)
        poor_image = self.reconstruct_image(poor_patches)
        
        # 调整尺寸确保输出尺寸正确
        rich_image = F.interpolate(rich_image, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        poor_image = F.interpolate(poor_image, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        
        # 通过SRM卷积处理
        rich_srm = self.srm_rich(rich_image)
        poor_srm = self.srm_poor(poor_image)
        
        # 通过ResNet提取特征
        rich_features = self.resnet_layers(rich_srm)
        poor_features = self.resnet_layers(poor_srm)
        
        # 平均池化并展平
        rich_features = rich_features.view(B, -1)
        poor_features = poor_features.view(B, -1)
        
        # 通过嵌入层获取最终特征
        rich_embedding = self.rich_embed(rich_features)
        poor_embedding = self.poor_embed(poor_features)
        
        return rich_embedding, poor_embedding

class EnhancedTextureAnalysisModule(nn.Module):
    """
    增强版纹理分析模块：结合了DCT和传统纹理分析方法
    """
    def __init__(self, patch_size=32, output_size=224, num_patches=9, dct_window_size=32):
        super(EnhancedTextureAnalysisModule, self).__init__()
        self.patch_size = patch_size
        self.output_size = output_size
        self.num_patches = num_patches
        self.dct_window_size = dct_window_size
        
        # DCT模块
        self.dct_module = DCT_base_Rec_Module(
            window_size=dct_window_size,
            stride=dct_window_size//2,
            output=output_size,
            grade_N=6,
            level_fliter=[0]
        )
        
        # SRM卷积层
        self.srm_rich = SRMConv()
        self.srm_poor = SRMConv()
        
        # ResNet50主干网络
        resnet = models.resnet50(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-1])
        
        # 特征融合和嵌入层
        self.fusion_layer = nn.Sequential(
            nn.Linear(2048*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)
        rich_textures = []
        poor_textures = []
        
        # 对批次中的每张图像分别处理
        for i in range(batch_size):
            # 提取单张图像
            img = x[i]  # [C, H, W]
            
            # 使用DCT模块提取丰富和贫乏纹理区域
            poor_texture, rich_texture, _, _ = self.dct_module(img)
            
            # 扩展维度并调整大小
            if poor_texture.size(0) < 3:
                poor_texture = poor_texture.repeat(3, 1, 1)
            if rich_texture.size(0) < 3:
                rich_texture = rich_texture.repeat(3, 1, 1)
                
            # 调整为期望的输出尺寸
            poor_texture = F.interpolate(
                poor_texture.unsqueeze(0), 
                size=(self.output_size, self.output_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            rich_texture = F.interpolate(
                rich_texture.unsqueeze(0), 
                size=(self.output_size, self.output_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            rich_textures.append(rich_texture)
            poor_textures.append(poor_texture)
        
        # 合并批次结果
        rich_texture_batch = torch.cat(rich_textures, dim=0)
        poor_texture_batch = torch.cat(poor_textures, dim=0)
        
        # 通过SRM卷积处理
        rich_srm = self.srm_rich(rich_texture_batch)
        poor_srm = self.srm_poor(poor_texture_batch)
        
        # 通过ResNet提取特征
        rich_features = self.resnet_layers(rich_srm).view(batch_size, -1)
        poor_features = self.resnet_layers(poor_srm).view(batch_size, -1)
        
        # 融合特征
        combined_features = torch.cat([rich_features, poor_features], dim=1)
        final_embedding = self.fusion_layer(combined_features)
        
        return final_embedding, rich_features, poor_features
