# models/mstd.py
# MSTD主模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_module import AdaptivePatchSelector, PatchShuffleModule
from .texture_module import TextureContrastModule
from .fusion_module import FeatureFusionModule

class MSTD(nn.Module):
    """
    多尺度语义-纹理融合检测器
    集成了三种不同的特征提取方式：
    1. 全局语义特征
    2. 多尺度patch shuffle特征
    3. 纹理对比度特征
    4. 频率域特征
    """
    def __init__(self, args):
        super(MSTD, self).__init__()
        self.args = args
        
        # 初始化CLIP视觉编码器，用于提取语义特征
        self.clip_encoder = self._init_clip_encoder(args.base_model)
        
        # 自适应patch选择模块
        self.patch_selector = AdaptivePatchSelector(args.patch_sizes)
        
        # Patch操作模块
        self.patch_shuffle = PatchShuffleModule()
        
        # 高通滤波器 (来自SRM，用于提取频率特征)
        self.srm_filters = self._init_srm_filters()
        
        # 丰富-贫乏纹理对比模块
        self.texture_contrast = TextureContrastModule(
            patch_size=args.texture_patch_size,
            num_patches=args.num_texture_patches
        )
        
        # 特征融合模块
        self.feature_fusion = FeatureFusionModule(feature_dim=args.feature_dim)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(args.feature_dim, args.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.feature_dim // 2, args.feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(args.dropout / 2),
            nn.Linear(args.feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def _init_clip_encoder(self, model_name):
        """初始化CLIP视觉编码器"""
        try:
            import clip
            model, _ = clip.load(model_name, device=self.args.device)
            return model.visual
        except:
            # 如果无法导入CLIP，使用占位符
            print("警告: 无法加载CLIP模型，使用占位符替代")
            return nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 512)
            )
    
    def _init_srm_filters(self):
        """初始化SRM高通滤波器，用于提取频率域特征"""
        srm_filters = []
        
        # 第一类滤波器：边缘检测
        filter1 = torch.tensor([[-1, 2, -1], 
                               [2, -4, 2], 
                               [-1, 2, -1]], dtype=torch.float32)
        srm_filters.append(filter1.view(1, 1, 3, 3))
        
        # 第二类滤波器：水平检测
        filter2 = torch.tensor([[0, 0, 0], 
                               [1, -2, 1], 
                               [0, 0, 0]], dtype=torch.float32)
        srm_filters.append(filter2.view(1, 1, 3, 3))
        
        # 第三类滤波器：垂直检测
        filter3 = torch.tensor([[0, 1, 0], 
                               [0, -2, 0], 
                               [0, 1, 0]], dtype=torch.float32)
        srm_filters.append(filter3.view(1, 1, 3, 3))
        
        return nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, 
                      padding=1, bias=False, groups=3)
            for _ in range(len(srm_filters))
        ])
    
    def forward(self, x):
        """前向传播过程"""
        batch_size = x.size(0)
        
        # 1. 提取全局语义特征
        semantic_features = self.clip_encoder(x)
        
        # 2. 自适应patch shuffle生成多尺度特征
        patch_sizes = self.patch_selector(x)
        multi_scale_features = []
        
        for patch_size in patch_sizes:
            # 应用patch shuffle操作
            shuffled_x = self.patch_shuffle(x, patch_size)
            # 提取特征
            patch_features = self.clip_encoder(shuffled_x)
            multi_scale_features.append(patch_features)
        
        # 3. 纹理对比度特征
        rich_texture, poor_texture = self.texture_contrast.extract_texture_regions(x)
        texture_contrast_features = self.texture_contrast(rich_texture, poor_texture)
        
        # 4. 频率域特征
        freq_features = []
        for i, filter_module in enumerate(self.srm_filters):
            # 每个滤波器分别应用并提取特征
            srm_output = filter_module(x)
            freq_features.append(self._process_freq_features(srm_output))
        
        # 特征融合
        fused_features = self.feature_fusion(
            semantic_features, 
            multi_scale_features, 
            texture_contrast_features, 
            freq_features
        )
        
        # 最终分类
        output = self.classifier(fused_features)
        
        return output
    
    def _process_freq_features(self, srm_output):
        """处理SRM滤波器输出的频率特征"""
        # 全局平均池化获取特征
        features = F.adaptive_avg_pool2d(srm_output, (1, 1))
        features = features.view(features.size(0), -1)
        return features