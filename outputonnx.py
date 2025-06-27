
import torch.onnx
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from utils.utils import FlexibleNamespace
import torch
import torch.nn as nn
from model.raft_trt import RAFT
outf = '/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_scale_10_stack_16_50mW/'
# Create a ConfigParser object
tmp = FlexibleNamespace()
# if os.path.exists(os.path.join(args_eval.model_path, 'args.json')):
args_model = tmp.load_from_json(os.path.join(outf, 'args.json'))

# load the network
checkpoint_path = os.path.join(outf, 'model_latest.pth')
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = nn.DataParallel(RAFT(args_model))
    # model = SPyNet('https://download.openmmlab.com/mmediting/restorers/''basicvsr/spynet_20210409-c6c1bd09.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

# 定义导出路径
onnx_file_path = outf + '/DeepIE_tensorRT.onnx'
# 定义输入
input1 = torch.randn(13, 1, 512, 512).cuda()
# input2 = torch.randn(8, 1, 512, 512).cuda()
dummy_input = input1

# 导出模型
torch.onnx.export(model.module, 
                  dummy_input, 
                  onnx_file_path, 
                  export_params=True,  # 导出模型参数
                  opset_version=16,    # 使用 ONNX 版本 
                  do_constant_folding=True,  # 是否折叠常量
                  input_names=['input'],  # 输入张量的名称
                  output_names=['output'],  # 输出张量的名称
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # 支持动态批大小

print(f"模型已成功导出为 {onnx_file_path}")


