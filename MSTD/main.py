# main.py
# 主入口文件

import os
import torch

from config import get_config, save_config
from train import train
from evaluate import evaluate, load_model
from datasets.dataloader import get_dataloaders

def main():
    # 获取配置
    args = get_config()
    print(f"使用设备: {args.device}")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(args)
    
    # 根据模式运行
    if args.mode == 'train':
        print("开始训练MSTD模型...")
        model = train(args)
        # 保存配置
        save_config(args)
        print("训练完成！")
        
    elif args.mode in ['eval', 'test']:
        # 确定检查点路径
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            if not os.path.exists(checkpoint_path):
                print(f"未找到检查点文件: {checkpoint_path}")
                print("请先训练模型或指定有效的检查点路径")
                return
        
        print(f"加载模型检查点: {checkpoint_path}")
        model = load_model(args, checkpoint_path)
        
        # 选择评估数据集
        data_loader = val_loader if args.mode == 'eval' else test_loader
        dataset_name = "验证集" if args.mode == 'eval' else "测试集"
        
        print(f"在{dataset_name}上评估模型...")
        metrics = evaluate(model, data_loader, args)
        
        print(f"{dataset_name}评估结果:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    else:
        print(f"不支持的模式: {args.mode}")

if __name__ == "__main__":
    main()