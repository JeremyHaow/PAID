# 示例脚本：测试纹理分析模块的功能

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from models.texture_analysis import TextureAnalysisModule, EnhancedTextureAnalysisModule

def visualize_texture_analysis(image_path, output_dir="./output"):
    """可视化纹理分析结果"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    
    # 初始化模块
    texture_module = TextureAnalysisModule(patch_size=32, output_size=224, num_patches=9)
    enhanced_module = EnhancedTextureAnalysisModule(patch_size=32, output_size=224, num_patches=9)
    
    # 处理图像
    with torch.no_grad():
        # 基础版本
        rich_patches, poor_patches = texture_module.extract_and_sort_patches(img_tensor)
        rich_image = texture_module.reconstruct_image(rich_patches)
        poor_image = texture_module.reconstruct_image(poor_patches)
        
        # 增强版本
        texture_embedding, rich_features, poor_features = enhanced_module(img_tensor)
        
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Origin Image")
    plt.axis('off')
    
    # 丰富纹理区域
    plt.subplot(2, 2, 2)
    rich_img = rich_image[0].permute(1, 2, 0).cpu().numpy()
    rich_img = np.clip(rich_img, 0, 1)
    plt.imshow(rich_img)
    plt.title("rich_img")
    plt.axis('off')
    
    # 贫乏纹理区域
    plt.subplot(2, 2, 3)
    poor_img = poor_image[0].permute(1, 2, 0).cpu().numpy()
    poor_img = np.clip(poor_img, 0, 1)
    plt.imshow(poor_img)
    plt.title("poor_img")
    plt.axis('off')
    
    # DCT方法提取的纹理结果
    plt.subplot(2, 2, 4)
    plt.bar(['rich texture', 'poor texture'], [rich_features.mean().item(), poor_features.mean().item()])
    plt.title("Texture Feature Intensity")
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{output_dir}/texture_analysis.png")
    plt.close()
    
    print(f"结果已保存到 {output_dir}/texture_analysis.png")

if __name__ == "__main__":
    # 替换为您自己的图像路径
    image_path = "test.jpg"
    visualize_texture_analysis(image_path)
