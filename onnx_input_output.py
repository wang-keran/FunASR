import onnx
import os
import onnxruntime as ort

# 设置模型路径
#model_path = "/home/wangkeran/桌面/WENET/test-cpu-non-streaming-output/encoder.onnx"
#model_path = "/home/wangkeran/桌面/WENET/aishellout-streaming/encoder.onnx"
#model_path = "/home/wangkeran/桌面/WENET/aishellout-non-streaming/encoder.onnx"
#model_path = "/home/wangkeran/桌面/WENET/gpu_cpu_non_streaming/encoder_fp16.onnx"
#model_path = "/home/wangkeran/桌面/WENET/gpu_cpu_streaming/encoder_fp16.onnx"
#model_path = "/home/wangkeran/桌面/WENET/gpu_cpu_english_streaming/encoder_fp16.onnx"
model_path = "/home/wangkeran-v10-2503-hwe/.cache/modelscope/hub/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/model.onnx"
# model_path = "/home/wangkeran-v10-2503-hwe/桌面/code/FunASRModelRepo/model_repo_paraformer_large_offline_punc/encoder_offline/1/model.onnx"
print("模型路径：",model_path)

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"Error: The model file at {model_path} does not exist.")
else:
    # 加载模型
    try:
        model = onnx.load(model_path)

        # 查看输入信息
        print("Inputs:")
        for input in model.graph.input:
            data_type = input.type.tensor_type.elem_type
            shape = input.type.tensor_type.shape
            shape_str = ', '.join([str(dim.dim_value) for dim in shape.dim]) if shape.dim else 'Unknown shape'
            print(f"  Input Name: {input.name}")
            print(f"    Data Type: {onnx.TensorProto.DataType.Name(data_type)}")
            print(f"    Shape: {shape_str}")

        # 查看输出信息
        print("\nOutputs:")
        for output in model.graph.output:
            data_type = output.type.tensor_type.elem_type
            shape = output.type.tensor_type.shape
            shape_str = ', '.join([str(dim.dim_value) for dim in shape.dim]) if shape.dim else 'Unknown shape'
            print(f"  Output Name: {output.name}")
            print(f"    Data Type: {onnx.TensorProto.DataType.Name(data_type)}")
            print(f"    Shape: {shape_str}")

        
        session = ort.InferenceSession(model_path)
        # 打印输入信息
        for input in session.get_inputs():
            print("Input:", input.name, input.shape, input.type)

        # 打印输出信息
        for output in session.get_outputs():
            print("Output:", output.name, output.shape, output.type)


    except Exception as e:
        print(f"Error loading the ONNX model: {e}")