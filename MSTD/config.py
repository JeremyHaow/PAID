# config.py
# 配置参数处理

import os
import torch
import argparse
import json

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MSTD - 多尺度语义-纹理融合AI生成图像检测器')
    
    # 运行模式配置
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test'], 
                        help='运行模式: train (训练), eval (在验证集上评估), test (在测试集上评估)')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='要加载的模型检查点路径，如果不提供则自动选择best_model.pth')
    parser.add_argument('--config_file', type=str, default=None, 
                        help='JSON配置文件路径，用于加载预设参数')
    
    # 通用配置
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 数据配置
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--train_data_path', type=str, default="data/train", help='包含0_real和1_fake文件夹的训练数据路径')
    parser.add_argument('--val_data_path', type=str, default="data/val", help='包含0_real和1_fake文件夹的验证数据路径')
    parser.add_argument('--test_data_path', type=str, default="data/test", help='包含0_real和1_fake文件夹的测试数据路径')
    parser.add_argument('--image_size', type=int, default=224, help='输入图像尺寸')
    
    # 模型配置
    parser.add_argument('--base_model', type=str, default='ViT-L/14', help='CLIP模型类型')
    parser.add_argument('--patch_sizes', type=int, nargs='+', default=[28, 56, 112], help='候选patch尺寸列表')
    parser.add_argument('--feature_dim', type=int, default=1024, help='特征维度')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout比例')
    
    # 纹理分析配置
    parser.add_argument('--texture_patch_size', type=int, default=32, help='纹理分析的patch尺寸')
    parser.add_argument('--num_texture_patches', type=int, default=64, help='纹理分析的patch数量')
    parser.add_argument('--num_patches', type=int, default=9, help='纹理重构使用的patch数量')
    parser.add_argument('--dct_window_size', type=int, default=32, help='DCT变换的窗口大小')
    
    # 训练配置
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'step'],
                        help='学习率调度策略 (plateau|cosine|step)')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5, help='学习率调度器耐心值 (用于plateau)')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='学习率衰减因子')
    
    # 对抗训练配置
    parser.add_argument('--adv_epsilon', type=float, default=0.01, help='对抗扰动强度')
    parser.add_argument('--adv_alpha', type=float, default=0.5, help='对抗损失权重')
    
    # 路径配置
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志保存目录')
    parser.add_argument('--result_dir', type=str, default='results', help='结果保存目录')
    
    return parser.parse_args()

def get_config():
    """
    获取配置：解析命令行参数、从配置文件加载和设置设备
    
    返回:
        args: 配置参数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 从配置文件加载参数（如果提供）
    if args.config_file is not None and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
            # 更新参数，只更新配置文件中存在的参数
            for key, value in config_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"警告：配置文件中的参数 '{key}' 不被识别")
    
    # 设置设备
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建必要的目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    return args

def save_config(args, filepath=None):
    """
    保存配置到文件
    
    参数:
        args: 配置参数
        filepath: 保存路径，如果未提供则使用默认路径
    """
    if filepath is None:
        filepath = os.path.join(args.log_dir, 'run_config.json')
    
    # 将args对象转换为字典，但排除device，因为它不能被JSON序列化
    config_dict = {k: v for k, v in vars(args).items() if k != 'device'}
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)
        print(f"配置已保存到: {filepath}")

# 向后兼容，以便现有代码可以继续工作
class Config:
    def __init__(self, args=None):
        if args is None:
            args = get_config()
            
        # 将所有参数设置为类属性
        for key, value in vars(args).items():
            setattr(self, key, value)