
import torch
from torch import nn

def decode_sigmod(generated_numbers):
    front_numbers = torch.round((generated_numbers[:, :5] - 1/(2*35))/(1/35)) + 1 
    back_numbers = torch.round((generated_numbers[:, 5:] - 1/(2*12))/(1/12)) + 1
    return front_numbers,back_numbers
    

# 判别损失函数
def discriminator_loss(y_true, y_pred, generated_numbers):
    bce_loss = nn.BCELoss()(y_pred, y_true)

    # simoid 解码成彩票号
    front_numbers,back_numbers = decode_sigmod(generated_numbers)
    front_numbers -= 1
    back_numbers -= 1
    
    # 前区和后区号码不重复约束
    front_unique_loss = torch.sum((torch.nn.functional.one_hot(front_numbers.to(torch.int64) , num_classes=35).sum(dim=1) > 1.0).float())
    back_unique_loss = torch.sum((torch.nn.functional.one_hot(back_numbers.to(torch.int64) , num_classes=12).sum(dim=1) > 1.0).float())

    # 组合损失
    total_loss = 0
    total_loss += bce_loss
    total_loss += (front_unique_loss + back_unique_loss)

    return total_loss

