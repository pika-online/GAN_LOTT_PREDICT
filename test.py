import torch
import numpy as np
from generator import Generator

# 定义生成器输入（噪声）的维度和生成器输出（彩票号码）的维度
input_dim = 100
output_dim = 7

# 实例化生成器模型
generator = Generator(input_dim, output_dim)

# 加载训练好的生成器模型权重
generator.load_state_dict(torch.load('generator_model.pth'))
generator.eval()  # 设置生成器为评估模式

# 生成新的噪声数据
batch_size = 10  # 生成10组彩票号码
noise = torch.randn(batch_size, input_dim)

# 通过生成器生成彩票号码
with torch.no_grad():  # 禁用梯度计算
    generated_numbers = generator(noise).numpy()

# 对生成的彩票号码进行后处理（反归一化）
# 假设前区号码范围1-35，后区号码范围1-12
front_area_numbers = generated_numbers[:, :5] * 34 + 1
back_area_numbers = generated_numbers[:, 5:] * 11 + 1

# 将彩票号码转换为整数
front_area_numbers = front_area_numbers.astype(int)
back_area_numbers = back_area_numbers.astype(int)

# 打印生成的彩票号码
print("Generated Lottery Numbers:")
for i in range(batch_size):
    print(f"Front Area: {front_area_numbers[i]}, Back Area: {back_area_numbers[i]}")

# 保存生成的彩票号码到文件
np.savetxt('generated_lottery_numbers.txt', np.hstack((front_area_numbers, back_area_numbers)), fmt='%d', delimiter=',')
