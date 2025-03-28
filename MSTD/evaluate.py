# evaluate.py
# 评估脚本

import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from models.mstd import MSTD
from utils.metrics import DetectionMetrics
from datasets.dataloader import get_dataloaders
from config import get_config

def evaluate(model, data_loader, args, save_plots=True):
    """
    评估模型性能
    
    参数:
        model: 模型
        data_loader: 数据加载器
        args: 配置参数
        save_plots: 是否保存曲线图
        
    返回:
        metrics: 评估指标
    """
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for real_batch, fake_batch in tqdm(data_loader, desc="Evaluating"):
            real_imgs, real_labels = real_batch
            fake_imgs, fake_labels = fake_batch
            
            real_imgs, real_labels = real_imgs.to(args.device), real_labels.to(args.device)
            fake_imgs, fake_labels = fake_imgs.to(args.device), fake_labels.to(args.device)
            
            # 合并图像和标签
            all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
            batch_labels = torch.cat([real_labels, fake_labels], dim=0)
            
            # 前向传播
            outputs = model(all_imgs)
            
            # 保存预测结果
            all_outputs.append(outputs)
            all_labels.append(batch_labels)
    
    # 连接所有预测结果和标签
    all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    
    # 计算评估指标
    metrics = DetectionMetrics.calculate_metrics(all_outputs, all_labels)
    
    if save_plots:
        # 创建保存结果的目录
        os.makedirs(args.result_dir, exist_ok=True)
        
        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(all_labels, all_outputs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(args.result_dir, 'roc_curve.png'))
        plt.close()
        
        # 绘制PR曲线
        precision, recall, _ = precision_recall_curve(all_labels, all_outputs)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(args.result_dir, 'pr_curve.png'))
        plt.close()
        
        # 保存详细评估结果
        with open(os.path.join(args.result_dir, 'evaluation_results.txt'), 'w') as f:
            f.write("评估结果：\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
    
    return metrics

def load_model(args, checkpoint_path):
    """
    加载预训练模型
    
    参数:
        args: 配置参数
        checkpoint_path: 模型检查点路径
        
    返回:
        model: 加载的模型
    """
    model = MSTD(args).to(args.device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def main():
    """主函数，进行模型评估"""
    # 获取配置
    args = get_config()
    print(f"使用设备: {args.device}")
    
    # 强制设置为评估模式
    args.mode = 'test'
    
    # 获取测试数据加载器
    _, _, test_loader = get_dataloaders(args)
    
    # 加载最佳模型
    model = load_model(args, os.path.join(args.checkpoint_dir, 'best_model.pth'))
    
    # 评估模型
    metrics = evaluate(model, test_loader, args)
    
    # 打印评估结果
    print("测试集评估结果：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()