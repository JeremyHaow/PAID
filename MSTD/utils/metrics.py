# utils/metrics.py
# 评估指标

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class DetectionMetrics:
    """AI生成图像检测评估指标"""
    
    @staticmethod
    def calculate_metrics(predictions, targets):
        """
        计算多种评估指标
        
        参数:
            predictions: 模型预测值 (0-1)
            targets: 真实标签 (0=真实, 1=AI生成)
            
        返回:
            metrics: 包含多种指标的字典
        """
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        # 二值化预测值
        binary_preds = (predictions > 0.5).astype(np.int32)
        
        # 计算指标
        accuracy = accuracy_score(targets, binary_preds)
        precision = precision_score(targets, binary_preds)
        recall = recall_score(targets, binary_preds)
        f1 = f1_score(targets, binary_preds)
        
        # 计算AUC
        try:
            auc = roc_auc_score(targets, predictions)
        except:
            auc = 0.5
            
        # 计算真实样本和AI生成样本的准确率
        real_indices = np.where(targets == 0)[0]
        fake_indices = np.where(targets == 1)[0]
        
        real_acc = accuracy_score(targets[real_indices], binary_preds[real_indices]) if len(real_indices) > 0 else 0
        fake_acc = accuracy_score(targets[fake_indices], binary_preds[fake_indices]) if len(fake_indices) > 0 else 0
        
        # 返回指标字典
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'real_acc': real_acc,
            'fake_acc': fake_acc
        }
        
        return metrics