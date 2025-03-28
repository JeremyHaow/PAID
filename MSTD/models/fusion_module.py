# models/fusion_module.py
# 特征融合模块

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionModule(nn.Module):
    """
    特征融合模块，采用注意力机制动态调整不同特征的权重
    这是MSTD的创新点之一
    """
    def __init__(self, feature_dim=1024):
        super(FeatureFusionModule, self).__init__()
        
        # 特征转换层
        self.semantic_transform = nn.Linear(512, 256)
        self.multi_scale_transform = nn.Linear(512, 256)
        self.texture_transform = nn.Linear(128, 256)
        self.freq_transform = nn.Linear(9, 256)
        
        # 动态注意力机制
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # 对应4种特征
            nn.Softmax(dim=1)
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, semantic_features, multi_scale_features, texture_contrast_features, freq_features):
        # 特征转换
        semantic_transformed = self.semantic_transform(semantic_features)
        
        # 处理多尺度特征
        if len(multi_scale_features) > 0:
            multi_scale_concat = torch.cat([feat.unsqueeze(1) for feat in multi_scale_features], dim=1)
            multi_scale_mean = multi_scale_concat.mean(dim=1)
            multi_scale_transformed = self.multi_scale_transform(multi_scale_mean)
        else:
            multi_scale_transformed = torch.zeros_like(semantic_transformed)
        
        # 处理纹理对比度特征
        texture_transformed = self.texture_transform(texture_contrast_features)
        
        # 处理频率域特征
        freq_concat = torch.cat([feat.unsqueeze(1) for feat in freq_features], dim=1)
        freq_mean = freq_concat.mean(dim=1)
        freq_transformed = self.freq_transform(freq_mean)
        
        # 特征串联
        all_features = torch.cat([
            semantic_transformed, 
            multi_scale_transformed, 
            texture_transformed, 
            freq_transformed
        ], dim=1)
        
        # 计算注意力权重
        attention_weights = self.attention(all_features)
        
        # 分离各个特征
        feature_chunks = torch.chunk(all_features, 4, dim=1)
        
        # 加权合并特征
        weighted_features = torch.zeros_like(all_features)
        for i, chunk in enumerate(feature_chunks):
            weighted_features += attention_weights[:, i:i+1] * chunk
        
        # 特征融合
        fused_features = self.fusion(weighted_features)
        
        return fused_features