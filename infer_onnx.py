import onnxruntime as ort
import numpy as np

def decode_sigmod(generated_numbers):
    front_numbers = np.round((generated_numbers[:, :5] - 1/(2*35))/(1/35)) + 1 
    back_numbers = np.round((generated_numbers[:, 5:] - 1/(2*12))/(1/12)) + 1
    return front_numbers, back_numbers

def filter_unique_numbers(front_numbers, back_numbers):
    unique_front_numbers = []
    unique_back_numbers = []

    for front, back in zip(front_numbers, back_numbers):
        if len(set(front)) == len(front) and len(set(back)) == len(back):
            unique_front_numbers.append(front)
            unique_back_numbers.append(back)

    return np.array(unique_front_numbers), np.array(unique_back_numbers)

# 加载 ONNX 模型
ort_session = ort.InferenceSession('models/generator_model.onnx')

# 定义输入字典
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# 进行推理
input_dim = 100
output_dim = 7
batch_size = 10

input_noise = np.random.randn(batch_size, input_dim).astype(np.float32)
generated_numbers = ort_session.run([output_name], {input_name: input_noise})[0]

# 解码生成的号码
front_numbers, back_numbers = decode_sigmod(generated_numbers)

# 筛选重复号码
unique_front_numbers, unique_back_numbers = filter_unique_numbers(front_numbers, back_numbers)

# 输出筛选后的号码
for a, b in zip(unique_front_numbers, unique_back_numbers):
    print(f"Front numbers: {a}, Back numbers: {b}")
