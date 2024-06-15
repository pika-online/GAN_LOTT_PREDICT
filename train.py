import numpy as np
import torch
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

from generator import Generator
from discriminator import Discriminator
from losses import discriminator_loss
from data_loader import lotto_data_loader

# 加载训练和验证数据
val_n = 100  # 验证集的数量
train_loader, val_loader = lotto_data_loader('./dataset/dlt_results.xlsx', val_n=val_n)

# 定义生成器和判别器的输入输出维度
input_dim = 100  # 生成器输入（噪声）的维度
output_dim = 7   # 生成器输出（彩票号码）的维度

# 实例化生成器和判别器模型
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# 为生成器和判别器定义优化器
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 设置损失和准确率的历史记录
window_size = 10
d_acc_history = deque(maxlen=window_size)  # 判别器准确率历史
g_loss_history = deque(maxlen=window_size)  # 生成器损失历史
d_loss_history = deque(maxlen=window_size)  # 判别器损失历史

# 存储用于绘图的历史记录
epochs_list = []
d_acc_values = []
g_loss_values = []
d_loss_values = []

# 定义训练参数
epochs = 10000
n_critic = 5  # 每次生成器更新对应的判别器更新次数

def evaluate(generator, discriminator, val_loader):
    """在验证集上评估模型的性能"""
    generator.eval()  # 设置生成器为评估模式
    discriminator.eval()  # 设置判别器为评估模式
    val_g_loss = 0.0
    val_d_loss = 0.0
    val_d_acc = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for real_numbers_batch in val_loader:
            real_numbers = real_numbers_batch[0]
            batch_size = real_numbers.size(0)

            # 为真实和生成的数据设置标签
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)

            # 从随机噪声生成假的彩票号码
            noise = torch.randn(batch_size, input_dim)
            fake_numbers = generator(noise)

            # 判别器损失
            real_loss = discriminator_loss(valid, discriminator(real_numbers), real_numbers)
            fake_loss = discriminator_loss(fake, discriminator(fake_numbers), fake_numbers)
            d_loss = 0.5 * (real_loss + fake_loss)
            val_d_loss += d_loss.item()

            # 生成器损失
            g_loss = discriminator_loss(valid, discriminator(fake_numbers), fake_numbers)
            val_g_loss += g_loss.item()

            # 判别器准确率
            real_acc = (discriminator(real_numbers) > 0.5).float().mean()
            fake_acc = (discriminator(fake_numbers) < 0.5).float().mean()
            d_acc = 0.5 * (real_acc + fake_acc)
            val_d_acc += d_acc.item()

    val_g_loss /= len(val_loader)
    val_d_loss /= len(val_loader)
    val_d_acc /= len(val_loader)
    
    generator.train()  # 重新设置生成器为训练模式
    discriminator.train()  # 重新设置判别器为训练模式
    return val_g_loss, val_d_loss, val_d_acc

# 训练循环
for epoch in range(epochs):
    for real_numbers_batch in train_loader:
        real_numbers = real_numbers_batch[0]
        batch_size = real_numbers.size(0)
        # print(real_numbers.shape)
        
        # 更新判别器n_critic次
        for _ in range(n_critic):
            # 为真实和生成的数据设置标签
            valid = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)

            # 从随机噪声生成假的彩票号码
            noise = torch.randn(batch_size, input_dim)
            fake_numbers = generator(noise)
            
            # 训练判别器
            optimizer_D.zero_grad()
            real_loss = discriminator_loss(valid, discriminator(real_numbers), real_numbers)
            fake_loss = discriminator_loss(fake, discriminator(fake_numbers.detach()), fake_numbers.detach())
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()
            
            # 计算判别器准确率
            real_acc = (discriminator(real_numbers) > 0.5).float().mean()
            fake_acc = (discriminator(fake_numbers.detach()) < 0.5).float().mean()
            d_acc = 0.5 * (real_acc + fake_acc)
        
        # 训练生成器
        optimizer_G.zero_grad()
        g_loss = discriminator_loss(valid, discriminator(fake_numbers), fake_numbers)
        g_loss.backward()
        optimizer_G.step()

    # 使用验证数据进行评估
    val_g_loss, val_d_loss, val_d_acc = evaluate(generator, discriminator, val_loader)

    # 记录基于验证数据的损失和准确率历史
    d_acc_history.append(val_d_acc)
    g_loss_history.append(val_g_loss)
    d_loss_history.append(val_d_loss)

    # 计算用于监控的统计数据，使用验证数据
    total_loss = val_g_loss + val_d_loss
    d_acc_mean = np.mean(d_acc_history)
    d_acc_variance = np.var(d_acc_history)
    g_loss_variance = np.var(g_loss_history)
    print(f"Epoch {epoch} [D loss: {val_d_loss} | D accuracy: {val_d_acc}] [G loss: {val_g_loss}] [G+D loss: {total_loss}]")
    print(f"d_acc_mean: {d_acc_mean}, d_acc_variance: {d_acc_variance}, g_loss_variance: {g_loss_variance}")

    # 存储用于绘图的历史记录
    epochs_list.append(epoch)
    d_acc_values.append(val_d_acc)
    g_loss_values.append(val_g_loss)
    d_loss_values.append(val_d_loss)

    # 早停条件
    if d_acc_variance < 0.1 and g_loss_variance < 0.1 and d_acc_mean < 0.55:
        print("D accuracy and G loss are stable. Stopping training.")
        torch.save(generator.state_dict(), 'models/generator_model.pth')
        torch.save(discriminator.state_dict(), 'models/discriminator_model.pth')
        break

# 绘制结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_list, d_acc_values, label='Discriminator Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Discriminator Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(epochs_list, g_loss_values, label='Generator Loss')
plt.plot(epochs_list, d_loss_values, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Generator and Discriminator Loss over Epochs')

plt.tight_layout()
plt.savefig('img/training_curves.png')
plt.show()
