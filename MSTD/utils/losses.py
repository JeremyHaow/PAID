# utils/losses.py
# 损失函数定义

import torch
import torch.nn as nn
import torch.nn.functional as F

class AIGCDetectionLoss(nn.Module):
    """
    AI生成图像检测的损失函数
    结合了分类损失和特征一致性损失
    """
    def __init__(self, consistency_weight=0.1):
        super(AIGCDetectionLoss, self).__init__()
        self.consistency_weight = consistency_weight
    
    def forward(self, outputs, labels, semantic_features, texture_features):
        """
        计算总损失
        
        参数:
            outputs: 模型输出的概率
            labels: 真实标签 (0=真实, 1=AI生成)
            semantic_features: 语义特征
            texture_features: 纹理特征
        
        返回:
            loss: 总损失
        """
        # 分类损失
        classification_loss = F.binary_cross_entropy(outputs, labels)
        
        # 特征一致性损失 (仅对真实图像)
        real_mask = (labels == 0).float().view(-1)
        if real_mask.sum() > 0:  # 确保至少有一个真实样本
            real_semantic = semantic_features[real_mask.bool()]
            real_texture = texture_features[real_mask.bool()]
            
            # 计算特征之间的一致性
            semantic_norm = F.normalize(real_semantic, p=2, dim=1)
            texture_norm = F.normalize(real_texture, p=2, dim=1)
            
            # 使用余弦相似度
            consistency_loss = 1 - (semantic_norm * texture_norm).sum(dim=1).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=outputs.device)
        
        # 总损失
        total_loss = classification_loss + self.consistency_weight * consistency_loss
        
        return total_loss, classification_loss, consistency_loss