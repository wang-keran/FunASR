from tritonclient.utils import np_to_triton_dtype, InferenceServerException
import numpy as np
import math
import soundfile as sf
import time
from funasr import AutoModel
import tritonclient.grpc as grpcclient


# 连接到 Triton Server（默认 gRPC 端口是 8001）
client = grpcclient.InferenceServerClient(url="localhost:8001")
client.load_model("punc")       # 这里两个模型先后顺序无所谓，都能正常加载起来，只要在发送请求前加载模型即可
client.load_model("use_punc")
# client.load_model("punc")

# 构造输入数据
input_text = ["你好世界","希望比失望更多人才能活得下去","为什么呢"]  # 可以传入多个句子，batch_size <= 64
batch_size = len(input_text)

# 确保输入是二维数组：shape = [batch_size, 1]
input_array = np.array([[text] for text in input_text], dtype=object)

# 创建输入张量（字符串类型需用 object 和 bytes）
input_tensor = grpcclient.InferInput("text_no_punc", [batch_size,1], "BYTES")
input_tensor.set_data_from_numpy(input_array)

# 设置输出张量
output_tensor = grpcclient.InferRequestedOutput("OUTPUT0")

# 发送推理请求
try:
    response = client.infer(
        model_name="use_punc",
        model_version="",  # 默认使用最新版本
        inputs=[input_tensor],
        outputs=[output_tensor],
        timeout=10000  # 超时时间（毫秒）
    )
    
    # 获取并解析输出结果
    output_data = response.as_numpy("OUTPUT0")[0]
    if type(output_data) == np.ndarray:
        result = b" ".join(output_data).decode("utf-8")
        # print("走了ndarray")
    else:
        result = output_data.decode("utf-8")
        # print("走了普通的decode")
    # print("output_data =", output_data)
    # print("dtype:", output_data.dtype)
    # print("shape:", output_data.shape)
    # print("type of first element:", type(output_data[0][0]))
    # output_text = [x[0].decode("utf-8") for x in output_data]
    print("模型输出:", result)

except InferenceServerException as e:
    print("Triton 推理错误:", e)