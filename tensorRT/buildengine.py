
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import onnx
# import onnx_tensorrt.backend as backend

# test = onnx.load('/home/shuran/RAFTCADSUN/checkpt/RAFTCAD_result_multiscale_stack_2002RAFTCAD_raw.onnx')
import tensorrt as trt

def build_engine(onnx_file_path, trt_model_path, max_workspace_size=1 << 36, fp16_mode=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()

    # 根据您的模型输入名称和形状进行调整
    input_name = "input"  
    min_shape = (13, 1, 512, 512)
    opt_shape = (13, 1, 512, 512)
    max_shape = (13, 1, 512, 512)
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build serialized engine.")
        return None

    with open(trt_model_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved as {trt_model_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine



# 示例调用
onnx_model_path = '/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_scale_10_stack_16_50mW/DeepIE_tensorRT.onnx'
trt_model_path = '/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_scale_10_stack_16_50mW/DeepIE_tensorRT.trt'
engine = build_engine(onnx_model_path, trt_model_path, fp16_mode=True)





