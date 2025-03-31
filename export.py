# method2, inference from local path
from funasr import AutoModel

model = AutoModel(
    #model="iic/emotion2vec_base",
    model="paraformer-zh-streaming",
    hub="ms"    #从model scope下载模型
)

res = model.export(type="onnx", quantize=False, opset_version=13, device='cpu')  # fp32 onnx-gpu
# res = model.export(type="onnx_fp16", quantize=False, opset_version=13, device='cuda')  # fp16 onnx-gpu
