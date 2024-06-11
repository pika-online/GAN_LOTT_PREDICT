import torch
import torch.nn as nn
import torch.onnx

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        # 定义模型的全连接层和激活函数
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),  # 输入层，输入维度为 input_dim，输出维度为512
            nn.LeakyReLU(0.2),  # LeakyReLU 激活函数
            nn.Linear(512, 256),  # 隐藏层，输入维度512，输出维度256
            nn.LeakyReLU(0.2),  # LeakyReLU 激活函数
            nn.Linear(256, 1),  # 输出层，输入维度256，输出维度1
            nn.Sigmoid()  # Sigmoid 激活函数，输出值在0到1之间
        )
    
    def forward(self, x):
        return self.model(x)  # 前向传播函数，返回模型的输出

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
        # print(self)  # 可以取消注释来打印整个模型结构
        print("\nLayers:")
        for name, layer in self.named_children():  # 遍历模型的每一层
            print(f"{name}: {layer}")

if __name__ == "__main__":
    input_dim = 100  # 输入数据的维度
    discriminator = Discriminator(input_dim)

    batch_size = 10  # 批量大小
    data = torch.rand(batch_size, input_dim)  # 随机生成输入数据

    output = discriminator(data)  # 前向传播，生成输出

    # 打印模型结构
    discriminator.print_model_structure()

    # 导出模型为 ONNX 格式
    discriminator.export_onnx("discriminator.onnx", (batch_size, input_dim))
