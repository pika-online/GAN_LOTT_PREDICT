import torch
import torch.nn as nn
import torch.onnx

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  # 输入层，输入维度为 input_dim，输出维度为256
            nn.LeakyReLU(0.2),  # LeakyReLU 激活函数
            nn.BatchNorm1d(256),  # 批量归一化
            nn.Linear(256, 512),  # 隐藏层，输入维度256，输出维度512
            nn.LeakyReLU(0.2),  # LeakyReLU 激活函数
            nn.BatchNorm1d(512),  # 批量归一化
            nn.Linear(512, 1024),  # 隐藏层，输入维度512，输出维度1024
            nn.LeakyReLU(0.2),  # LeakyReLU 激活函数
            nn.BatchNorm1d(1024),  # 批量归一化
            nn.Linear(1024, output_dim),  # 输出层，输入维度1024，输出维度 output_dim
            nn.Sigmoid()  # Sigmoid 激活函数，输出值在0到1之间
        )
    
    def forward(self, x):
        """前向传播函数"""
        return self.model(x)  # 将数据传入模型，得到输出，形状为 N x output_dim

    def export_onnx(self, file_path, input_size):
        # 创建模型的虚拟输入
        dummy_input = torch.randn(*input_size)
        # 导出模型为 ONNX 格式
        torch.onnx.export(self, 
                          dummy_input, 
                          file_path, 
                          export_params=True,  # 是否导出模型参数
                          opset_version=11,  # ONNX opset 版本
                          do_constant_folding=True,  # 是否执行常量折叠优化
                          input_names=['input'],  # 输入名称
                          output_names=['output'],  # 输出名称
                          dynamic_axes={'input': {0: 'batch_size'},  # 动态轴，支持可变批量大小
                                        'output': {0: 'batch_size'}})

    def print_model_structure(self):
        print("Model structure:")
        print("\nLayers:")
        for name, layer in self.named_children():  # 遍历模型的每一层
            print(f"{name}: {layer}")

if __name__ == "__main__":
    input_dim = 100  # 输入数据的维度
    output_dim = 7  # 输出数据的维度
    generator = Generator(input_dim, output_dim)

    batch_size = 10  # 批量大小
    noise = torch.rand(batch_size, input_dim)  # 随机生成输入数据

    output = generator(noise)  # 前向传播，生成输出

    # 打印模型结构
    generator.print_model_structure()

    # 导出模型为 ONNX 格式
    generator.export_onnx("models/generator.onnx", (batch_size, input_dim))
