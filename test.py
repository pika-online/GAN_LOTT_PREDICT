import torch
import numpy as np
import matplotlib.pyplot as plt
from generator import Generator

# 定义生成器输入（噪声）的维度和生成器输出（彩票号码）的维度
input_dim = 100
output_dim = 7

# 实例化生成器模型
generator = Generator(input_dim, output_dim)

# 加载训练好的生成器模型权重
generator.load_state_dict(torch.load('models/generator_model.pth'))
generator.eval()  # 设置生成器为评估模式

# 生成新的噪声数据
batch_size = 300  # 生成100组彩票号码
noise = torch.randn(batch_size, input_dim)

# 通过生成器生成彩票号码
with torch.no_grad():  # 禁用梯度计算
    generated_numbers = generator(noise).numpy()

# 对生成的彩票号码进行后处理（反归一化）
# 假设前区号码范围1-35，后区号码范围1-12
front_area_numbers = generated_numbers[:, :5] * 35 
back_area_numbers = generated_numbers[:, 5:] * 12 

# 将彩票号码转换为四舍五入的整数
front_area_numbers = np.floor(front_area_numbers) + 1
back_area_numbers = np.floor(back_area_numbers) + 1

# 打印生成的彩票号码
print("Generated Lottery Numbers:")
for i in range(batch_size):
    print(f"Front Area: {front_area_numbers[i]}, Back Area: {back_area_numbers[i]}")

# 保存生成的彩票号码到文件
np.savetxt('generated_lottery_numbers.txt', np.hstack((front_area_numbers, back_area_numbers)), fmt='%d', delimiter=',')

# 绘制前区号码的分布
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.hist(front_area_numbers.flatten(), bins=np.arange(1, 37)-0.5, edgecolor='black')
plt.xlabel('Front Area Number')
plt.ylabel('Frequency')
plt.title('Distribution of Front Area Numbers')

# 绘制后区号码的分布
plt.subplot(1, 2, 2)
plt.hist(back_area_numbers.flatten(), bins=np.arange(1, 14)-0.5, edgecolor='black')
plt.xlabel('Back Area Number')
plt.ylabel('Frequency')
plt.title('Distribution of Back Area Numbers')

plt.tight_layout()
plt.savefig('img/lotto_numbers_distribution_predict.png')
plt.show()
