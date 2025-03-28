# train.py
# 训练脚本

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.mstd import MSTD
from utils.adversarial import AdversarialTrainer
from utils.losses import AIGCDetectionLoss
from utils.metrics import DetectionMetrics
from datasets.dataloader import get_dataloaders
from config import get_config

def train(args):
    """
    训练MSTD模型
    
    参数:
        args: 配置参数
    """
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 初始化tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 获取数据加载器
    train_loader, val_loader, _ = get_dataloaders(args)
    
    # 创建模型
    model = MSTD(args).to(args.device)
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    if args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            verbose=True
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs,
            eta_min=1e-6
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.num_epochs // 3,
            gamma=0.1
        )
    
    # 创建损失函数
    criterion = AIGCDetectionLoss()
    
    # 创建对抗训练工具
    adv_trainer = AdversarialTrainer(model, optimizer, args)
    
    # 记录最佳验证指标
    best_val_auc = 0.0
    
    # 训练循环
    for epoch in range(args.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'real_acc': 0.0,
            'fake_acc': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for real_batch, fake_batch in pbar:
            real_imgs, real_labels = real_batch
            fake_imgs, fake_labels = fake_batch
            
            real_imgs, real_labels = real_imgs.to(args.device), real_labels.to(args.device)
            fake_imgs, fake_labels = fake_imgs.to(args.device), fake_labels.to(args.device)
            
            # 对抗训练步骤
            loss = adv_trainer.train_step(real_imgs, fake_imgs)
            train_loss += loss
            
            # 更新进度条
            pbar.set_postfix(loss=f"{loss:.4f}")
            
            # 预测结果用于计算指标
            with torch.no_grad():
                all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
                all_labels = torch.cat([real_labels, fake_labels], dim=0)
                
                outputs = model(all_imgs)
                batch_metrics = DetectionMetrics.calculate_metrics(outputs, all_labels)
                
                # 更新累计指标
                for k in train_metrics:
                    train_metrics[k] += batch_metrics[k]
        
        # 计算平均值
        train_loss /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # 记录训练指标
        writer.add_scalar('Loss/train', train_loss, epoch)
        for k, v in train_metrics.items():
            writer.add_scalar(f'Metrics/{k}/train', v, epoch)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            for real_batch, fake_batch in pbar:
                real_imgs, real_labels = real_batch
                fake_imgs, fake_labels = fake_batch
                
                real_imgs, real_labels = real_imgs.to(args.device), real_labels.to(args.device)
                fake_imgs, fake_labels = fake_imgs.to(args.device), fake_labels.to(args.device)
                
                # 合并图像和标签
                all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
                batch_labels = torch.cat([real_labels, fake_labels], dim=0)
                
                # 前向传播
                outputs = model(all_imgs)
                
                # 计算损失
                # 注：这里简化处理，实际上需要提取语义特征和纹理特征
                batch_loss = torch.nn.functional.binary_cross_entropy(outputs, batch_labels)
                val_loss += batch_loss.item()
                
                # 保存预测结果
                all_outputs.append(outputs)
                all_labels.append(batch_labels)
                
                # 更新进度条
                pbar.set_postfix(loss=f"{batch_loss.item():.4f}")
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        
        # 连接所有预测结果和标签
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算验证指标
        val_metrics = DetectionMetrics.calculate_metrics(all_outputs, all_labels)
        
        # 记录验证指标
        writer.add_scalar('Loss/val', val_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Metrics/{k}/val', v, epoch)
        
        # 更新学习率
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 打印训练和验证指标
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"  Real Acc: {val_metrics['real_acc']:.4f}, Fake Acc: {val_metrics['fake_acc']:.4f}")
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"  保存最佳模型，AUC: {best_val_auc:.4f}")
        
        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
        }, os.path.join(args.checkpoint_dir, 'latest_model.pth'))
    
    # 关闭tensorboard writer
    writer.close()
    
    print(f"训练完成！最佳验证 AUC: {best_val_auc:.4f}")
    return model

if __name__ == "__main__":
    # 获取配置
    args = get_config()
    print(f"使用设备: {args.device}")
    
    # 开始训练
    model = train(args)