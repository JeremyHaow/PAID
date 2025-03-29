# datasets/dataloader.py
# 数据加载器

import os
import glob
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from .augmentation import get_augmentations

class AIGCDetectionDataset(Dataset):
    """
    AI生成图像检测数据集
    加载真实图像和AI生成的图像
    数据集结构:
    dataset_path/
        0_real/
            real_image1.jpg
            real_image2.jpg
            ...
        1_fake/
            fake_image1.jpg
            fake_image2.jpg
            ...
    """
    def __init__(self, dataset_path, transform=None, phase='train'):
        """
        初始化数据集
        
        参数:
            dataset_path: 数据集路径，包含0_real和1_fake子目录
            transform: 数据变换
            phase: train/val/test
        """
        self.phase = phase
        self.transform = transform
        
        # 定义真实和生成图像的路径
        real_dir = os.path.join(dataset_path, "0_real")
        fake_dir = os.path.join(dataset_path, "1_fake")
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            raise ValueError(f"目录结构不正确! 需要在{dataset_path}下有0_real和1_fake文件夹。")
        
        # 加载真实图像
        self.real_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.real_images.extend(glob.glob(os.path.join(real_dir, '**', ext), recursive=True))
            
        # 加载假图像
        self.fake_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.fake_images.extend(glob.glob(os.path.join(fake_dir, '**', ext), recursive=True))
        
        # 确保数据集均衡
        min_size = min(len(self.real_images), len(self.fake_images))
        if self.phase == 'train':
            # 训练集使用全部数据
            self.real_images = self.real_images[:min_size]
            self.fake_images = self.fake_images[:min_size]
        else:
            # 验证/测试集使用部分数据，如果数据较少可以全部使用
            val_size = min(min_size//5, min_size)
            self.real_images = self.real_images[:val_size]
            self.fake_images = self.fake_images[:val_size]
        
        print(f"{phase}数据集加载完成，共 {len(self.real_images)} 真实图像和 {len(self.fake_images)} AI生成图像")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.real_images) + len(self.fake_images)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 决定是获取真实图像还是假图像
        is_real = idx < len(self.real_images)
        
        if is_real:
            img_path = self.real_images[idx]
            label = 0  # 真实图像
        else:
            img_path = self.fake_images[idx - len(self.real_images)]
            label = 1  # AI生成图像
        
        # 加载图像
        try:
            img = Image.open(img_path).convert("RGB")
            
            # 应用变换
            if self.transform:
                img = self.transform(img)
                
            # 转换标签为张量
            label = torch.tensor(label, dtype=torch.float32).view(1)
            
            return img, label
            
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            # 返回随机图像
            return self.__getitem__(random.randint(0, self.__len__() - 1))


def get_dataloaders(args):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        args: 命令行参数
        
    返回:
        train_loader, val_loader, test_loader
    """
    # 获取数据增强
    train_transform, val_transform = get_augmentations(args.image_size)
    
    # 创建数据集
    train_dataset = AIGCDetectionDataset(
        args.train_data_path,
        transform=train_transform,
        phase='train'
    )
    
    val_dataset = AIGCDetectionDataset(
        args.val_data_path,
        transform=val_transform,
        phase='val'
    )
    
    test_dataset = AIGCDetectionDataset(
        args.test_data_path,
        transform=val_transform,
        phase='test'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader