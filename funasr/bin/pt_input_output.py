import torch

# 直接加载模型
model = torch.load("/home/wangkeran-v10-2503-hwe/桌面/code/sherpa-onnx/python-api-examples/sherpa-onnx-online-punct-en-2024-08-06/sherpa-onnx-online-punct-en-2024-08-06/model.onnx")
model.eval()

print(model)

# 你仍然可以通过 dummy input 测试输入输出
dummy_input = torch.randn(1, 3, 224, 224)  # 根据需要改
output = model(dummy_input)
print(f"输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")