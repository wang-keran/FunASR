import onnx
onnx_model = onnx.load("/home/wangkeran-v10-2503-hwe/桌面/code/FunASRModelRepo/model_repo_paraformer_large_offline/encoder/1/model_quant.onnx")  # 看能不能正常加载
onnx.checker.check_model(onnx_model)  # 检查是否符合标准