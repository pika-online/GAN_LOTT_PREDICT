import torch
from generator import Generator




# 定义生成器输入（噪声）的维度和生成器输出（彩票号码）的维度
input_dim = 100
output_dim = 7
batch_size = 10

# 实例化生成器模型
generator = Generator(input_dim, output_dim)

# 加载训练好的生成器模型权重
generator.load_state_dict(torch.load('models/generator_model.pth'))
generator.eval()  # 设置生成器为评估模式

# 导出模型为 ONNX 格式
generator.export_onnx('models/generator_model.onnx', (batch_size, input_dim))


