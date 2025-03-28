# utils/adversarial.py
# 对抗训练工具

import torch
import torch.nn.functional as F

class AdversarialTrainer:
    """
    对抗性训练工具，提高模型鲁棒性
    """
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        
    def train_step(self, real_images, fake_images):
        """
        执行一步对抗训练
        
        参数:
            real_images: 真实图像
            fake_images: AI生成的图像
        
        返回:
            loss: 训练损失
        """
        # 正常训练步骤
        self.optimizer.zero_grad()
        
        # 合并真假图像
        all_images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([
            torch.zeros(real_images.size(0), 1),
            torch.ones(fake_images.size(0), 1)
        ], dim=0).to(all_images.device)
        
        # 前向传播
        outputs = self.model(all_images)
        loss = F.binary_cross_entropy(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 对抗样本生成
        # 仅为假图像创建对抗样本（使它们更难被检测）
        fake_images.requires_grad = True
        fake_outputs = self.model(fake_images)
        fake_labels = torch.ones(fake_images.size(0), 1).to(fake_images.device)
        
        # 计算梯度
        fake_loss = F.binary_cross_entropy(fake_outputs, fake_labels)
        fake_loss.backward(retain_graph=True)
        
        # 生成对抗样本
        adv_fake_images = fake_images - self.args.adv_epsilon * fake_images.grad.sign()
        adv_fake_images = torch.clamp(adv_fake_images, 0, 1)
        
        # 对抗样本训练
        adv_outputs = self.model(adv_fake_images)
        adv_loss = F.binary_cross_entropy(adv_outputs, fake_labels)
        
        # 结合原始损失和对抗损失
        total_loss = loss + self.args.adv_alpha * adv_loss
        total_loss.backward()
        
        # 更新模型参数
        self.optimizer.step()
        
        return total_loss.item()