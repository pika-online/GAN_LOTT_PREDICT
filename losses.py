
import torch
from torch import nn

# 判别损失函数
def discriminator_loss(y_true, y_pred, generated_numbers):
    bce_loss = nn.BCELoss()(y_pred, y_true)

    # 获取生成的号码
    front_numbers = generated_numbers[:, :5] * 35   # 反归一化到1-35
    back_numbers = generated_numbers[:, 5:] * 12   # 反归一化到1-12

    # 前区和后区号码范围约束（仅最大值约束）
    front_range_loss = torch.sum(torch.clamp(front_numbers - 35.0, min=0.0))
    back_range_loss = torch.sum(torch.clamp(back_numbers - 12.0, min=0.0))

    front_numbers = torch.clamp(front_numbers,0,34)
    back_numbers = torch.clamp(back_numbers,0,11)
    # 前区和后区号码不重复约束
    front_unique_loss = torch.sum((torch.nn.functional.one_hot(front_numbers.to(torch.int64) , num_classes=35).sum(dim=1) > 1.0).float())
    back_unique_loss = torch.sum((torch.nn.functional.one_hot(back_numbers.to(torch.int64) , num_classes=12).sum(dim=1) > 1.0).float())

    # 组合损失
    total_loss = 0
    total_loss += bce_loss
    # total_loss += (front_range_loss + back_range_loss)
    total_loss += (front_unique_loss + back_unique_loss)

    return total_loss

