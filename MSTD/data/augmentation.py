# datasets/augmentation.py
# 数据增强方法

from torchvision import transforms

def get_augmentations(image_size):
    """
    创建训练和验证的数据增强
    
    参数:
        image_size: 目标图像尺寸
        
    返回:
        train_transform, val_transform
    """
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.1, 
                contrast=0.1, 
                saturation=0.1, 
                hue=0.1
            ),
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation(10),
        ], p=0.3),
        # 模拟常见图像处理
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ], p=0.2),
        # JPEG压缩模拟
        transforms.RandomApply([
            transforms.Lambda(lambda x: transforms.functional.adjust_sharpness(x, 0.5)),
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # 验证/测试集数据增强
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return train_transform, val_transform