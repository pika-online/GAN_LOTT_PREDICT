import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

def lotto_data_loader(file_path, val_n=50, seed=24, batch_size=32):
    # 读取文件
    data = pd.read_excel(file_path)

    # 数据预处理：拆分前区和后区号码并合并为一个数据集，然后进行归一化处理
    front_area_numbers = data['前区'].str.split(' ', expand=True).astype(int)
    back_area_numbers = data['后区'].str.split(' ', expand=True).astype(int)

    # 归一化处理
    front_area_numbers = (front_area_numbers - 1) / 34.0  # 前区号码范围1-35，归一化到0-1
    back_area_numbers = (back_area_numbers - 1) / 11.0   # 后区号码范围1-12，归一化到0-1

    # 合并前区和后区的号码
    all_numbers = pd.concat([front_area_numbers, back_area_numbers], axis=1).values
    all_numbers = torch.tensor(all_numbers, dtype=torch.float32)
    print(f"All numbers shape: {all_numbers.shape}")

    # Create TensorDataset
    dataset = TensorDataset(all_numbers)

    # 划分数据集
    data_size = len(all_numbers) - val_n
    train_dataset, val_dataset = random_split(dataset, 
                                              [data_size, val_n],
                                              generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader

def visualize_lotto_numbers(train_loader):
    # 收集所有彩票号码
    all_numbers = []
    for real_numbers_batch in train_loader:
        real_numbers = real_numbers_batch[0].numpy()
        all_numbers.append(real_numbers)
    all_numbers = np.concatenate(all_numbers, axis=0)

    # 反归一化
    front_area_numbers = all_numbers[:, :5] * 34 + 1
    back_area_numbers = all_numbers[:, 5:] * 11 + 1

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
    plt.savefig('lotto_numbers_distribution.png')
    plt.show()

if __name__ == "__main__":
    train_loader, val_loader = lotto_data_loader('./dlts.xlsx')
    for real_numbers_batch in train_loader:
        print(real_numbers_batch[0].shape)  # Print the shape of the actual tensor in the batch
    
    visualize_lotto_numbers(train_loader)
