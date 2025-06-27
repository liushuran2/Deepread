import pycuda.autoinit  # 自动初始化 CUDA 上下文
import pycuda.driver as cuda
import tensorrt as trt
import torch
import numpy as np
def load_engine(trt_engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER) # 创建TensorRT运行时对象
    try:
        with open(trt_engine_path,'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read()) # 反序列化引擎，将引擎从文件中加载到内存中，加载到引擎对象中
        print(f"成功加载引擎: {trt_engine_path}")
        return engine
    except Exception as e:
        print(f"Failed to deserialize the engine: {e}")
        return None


if __name__ == "__main__":  
    trt_path = '/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_scale_10_stack_28_50mW_fton10mW/DeepIE_tensorRT.trt'
    engine = load_engine(trt_path)
    if engine is None:
        print("加载引擎失败。")
    context = engine.create_execution_context()

    import time
    starttime = time.time()
    # 分配缓冲区
    input_data = np.random.randn(13, 1, 512, 512).astype(np.float32)
    output_data = np.empty([1, 1, 12, 512, 512], dtype=np.float32)

    # 分配CUDA内存
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)

    # 将输入数据复制到GPU
    cuda.memcpy_htod(d_input, input_data)
    # 执行推理
    bindings = [int(d_input), int(d_output)]
    context.execute_v2(bindings)

    # 将输出数据从GPU复制回主机
    cuda.memcpy_dtoh(output_data, d_output)

    endtime = time.time()
    print(f"tensorRT推理时间: {endtime-starttime:.3f}s")

    # # 释放CUDA内存
    # d_input.free()
    # d_output.free()
    # 释放引擎和上下文
    # context.destroy()
    # engine.destroy()
    print("推理完成。")

    # # Compare this with regular PyTorch inference
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    # import torch
    # import torch.nn as nn
    # import sys
    # from pathlib import Path

    # parent_dir = Path(__file__).resolve().parent.parent
    # sys.path.append(str(parent_dir))
    # from model.raft_trt import RAFT
    # from utils.utils import FlexibleNamespace
    # outf = '/home/shuran/RAFTCADSUN/checkpt/RAFTCAD_result_multiscale_stack_2002'
    # # Create a ConfigParser object
    # tmp = FlexibleNamespace()
    # # if os.path.exists(os.path.join(args_eval.model_path, 'args.json')):
    # args_model = tmp.load_from_json(os.path.join(outf, 'args.json'))
    # # load the network
    # checkpoint_path = os.path.join(outf, 'model_latest.pth')
    # if os.path.isfile(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #     model = nn.DataParallel(RAFT(args_model))
    #     # model = SPyNet('https://download.openmmlab.com/mmediting/restorers/''basicvsr/spynet_20210409-c6c1bd09.pth')
    #     model.load_state_dict(checkpoint['state_dict'])
    #     model.cuda()
    #     model.eval()

    # input1 = torch.randn(11, 1, 512, 512).cuda()

    # starttime = time.time()
    # output1 = model(input1)
    # endtime = time.time()
    # print(f"PyTorch推理时间: {endtime-starttime:.3f}s")

