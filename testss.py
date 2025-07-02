from __future__ import print_function, division
import sys
import multiprocessing as mp
import argparse
import configparser
import os
import queue
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
import yaml
import tifffile as tiff
import torch
import torch.nn as nn
# import tensorflow as tf
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader
import scipy
from scipy import io
# implemented
from model.raft import RAFT
from model.spynet import SPyNet
from dataset.datasets import *
from engine import *
from utils.frame_utils import *
from model.loss import sequence_loss
from suns.Network.shallow_unet import get_shallow_unet
from suns.Network.shallow_unet import ShallowUNet
from suns.Online.functions_init import init_online, plan_fft2
from suns.Online.functions_online import merge_2, merge_2_nocons, merge_complete, select_cons, \
    preprocess_online_batch, CNN_online_batch, separate_neuron_online_batch, refine_seperate_cons_online, final_merge, extrace_trace
from suns.PostProcessing.combine import segs_results, unique_neurons2_simp, \
    group_neurons, piece_neurons_IOU, piece_neurons_consume, convert_mask
from scipy.io import savemat, loadmat
from suns.PreProcessing.preprocessing_functions import preprocess_video, \
    plan_fft, plan_mask2, load_wisdom_txt, export_wisdom_txt, \
    SNR_normalization, median_normalization, median_calculation, find_dataset, preprocess_complete
from tensorRT.loadengine import load_engine
from processframes import process_frames_init

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # test file
    parser.add_argument('--model_path', default='/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_scale_10_stack_28_50mW_fton10mW/', help='path to the trained model')
    # parser.add_argument('--suns_model_path', default='/mnt/nas01/LSR/DATA/2p_bench/suns/suns_h5/0323/Weights/model_best4220.pth', help='path to the trained model')
    parser.add_argument('--suns_model_path', default='/mnt/nas01/LSR/DATA/2p_bench/suns/0701/Weights/model_latest.pth', help='path to the trained model')
    parser.add_argument('--gt_flow', type=str, nargs='+', default=None, 
                        help='test file for evaluation')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--add_blur', default=False, action='store_true')
    parser.add_argument('--doublestage', default=True, action='store_true')
    parser.add_argument('--update_baseline', default=True, action='store_true')
    parser.add_argument('--frames_init', type=int, default=150, help='buffer size of initialization')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of denoising')
    parser.add_argument('--overlap_size', type=int, default=2, help='batch size of denoising')
    parser.add_argument('--max_neuron_num', type=int, default=1200, help='batch size of denoising')

    # parser
    args_eval = parser.parse_args()

    # load the flow
    if args_eval.gt_flow is not None:
        args_eval.gt_flow = scipy.io.loadmat(args_eval.gt_flow[0])['motion_field']
    else:
        args_eval.gt_flow = None

    print(os.path.join(args_eval.model_path, 'args.json'))
    # Create a ConfigParser object
    tmp = FlexibleNamespace()
    # if os.path.exists(os.path.join(args_eval.model_path, 'args.json')):
    args_model = tmp.load_from_json(os.path.join(args_eval.model_path, 'args.json'))
    print_args_formatted(args_model)
    # else:
    #     print("No args.json file found.")
    #     sys.exit()

    # get the output path
    outf = args_eval.model_path
    
    # load SUNs CNN model
    # fff = get_shallow_unet()
    filename_CNN = args_eval.suns_model_path
    fff = nn.DataParallel(ShallowUNet(), device_ids=[0])
    fff.cuda()
    fff.eval
    checkpoint = torch.load(filename_CNN, map_location='cpu')
    fff.load_state_dict(checkpoint['state_dict'])

    # load the network
    import pycuda.autoinit  # This is needed to initialize the PyCUDA driver
    trt_path = '/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_scale_10_stack_28_50mW_fton10mW/DeepIE_tensorRT.trt'
    engine = load_engine(trt_path)
    if engine is None:
        print("加载引擎失败。")

    frames_init = args_eval.frames_init

    # get the evaluating dataset
    test_dataloader_array = []
    args_eval.data_property  = []
    args_eval.norm_type = args_model.norm_type
    update_baseline = args_eval.update_baseline

    Params_post={
            # minimum area of a neuron (unit: pixels).
            'minArea': 100, 
            # average area of a typical neuron (unit: pixels) 
            'avgArea': 180,
            # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
            'thresh_pmap': 130, 
            # values higher than "thresh_mask" times the maximum value of the mask are set to one.
            'thresh_mask': 0.4, 
            # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels)
            'thresh_COM0': 2, 
            # maximum COM distance of two masks to be considered the same neuron (unit: pixels)
            'thresh_COM': 6, 
            # minimum IoU of two masks to be considered the same neuron
            'thresh_IOU': 0.5, 
            # minimum consume ratio of two masks to be considered the same neuron 
            'thresh_consume': 0.7, 
            # minimum consecutive number of frames of active neurons
            'cons': 4}
    
    max_neuron_num = args_eval.max_neuron_num

    # root_path = '/mnt/nas02/LSR/DATA/Allen/ophys_tiff/'
    root_path = '/mnt/nas01/LSR/DATA/2p_bench/2p_148d/'
    # get the list of directories in the root path
    directories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    # directories = ['671162628']
    # directories = ['501271265', '501704220', '524691284', '531006860',  '603516552', '604145810', '607040613', '669233895', '671162628', '679353932']
    directories.sort()  # Sort directories by number

    for directory_id in directories:
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/NAOMi_2p_4/test_seq_6000/' + str(directory_id+1) + '/'
        # loadframe_path = directory_path
        # directory_path = '/mnt/nas01/LSR/DATA/2p_bench/2p_148d/trial_2p_' + str(directory_id + 1) + '/'
        # directory_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_test_long/result/scale_10/DeepIE/50mW/'
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/simulation_noise/50W_10scale_5range_gt/'
        # directory_path = '/mnt/nas01/LSR/DATA/2p_bench/HP01/HP01/'
        # directory_path = '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/ABO/ABO_2024/577313742/'
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/mini2p/ca1/' + str(directory_id+1) + '/'
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/2p_fiberscopy/' + str(directory_id+1)+ '/'
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/2p_benchtop/2p_148d/trial_2p_' + str(directory_id+1) + '/motion/'
        directory_path = os.path.join(root_path, directory_id) + '/'
        # directory_path = 'DATA/'
        # loadframe_path = directory_path + 'movie/'
        loadframe_path = directory_path
        all_files = os.listdir(loadframe_path)
    
        tiff_files = [filename for filename in all_files if filename.endswith('.tiff')]
        tiff_files.sort()

        save_path = directory_path + 'test/'
        p = mp.Pool()

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with torch.no_grad():
            for fnames in tiff_files:
                # fnames = 'Fsim_clean.tiff'
                session_name = fnames.split('/')[-1].split('.')[0]
                datapath = loadframe_path + fnames
                video_raw = tiff.imread(datapath).astype(np.float32)
                p1_val = np.percentile(video_raw, 1)
                p99_val = np.percentile(video_raw, 99)
                args_eval.data_property = {
                'p1': p1_val,
                'p99': p99_val,
                }

                video_adjust, template, Masks, traces = process_frames_init(video_raw, Params_post, args_eval.batch_size, args_eval.overlap_size, engine, fff, p)

                video_array = np.nan_to_num(video_adjust, nan=0)
                denorm_fun = lambda normalized_video: postprocessing_video(normalized_video, args_eval.norm_type, args_eval.data_property) 
                video_array = denorm_fun(video_array)

                # video_array = np.concatenate((video_adjust, video_array), axis=0).astype(np.float32)
 
                output_file = os.path.join(save_path, '{}_reg.tiff'.format(session_name))
                # save_image(video_array, str(data_property["data_type"]), output_file)
                save_image(video_array, 'uint16', output_file)

                savemat(os.path.join(save_path, '{}_Output_Masks.mat'.format(session_name)), \
            {'Masks':Masks}, do_compression=True)
                
                savemat(os.path.join(save_path, '{}_Extracted_trace.mat'.format(session_name)), \
            {'trace':traces}, do_compression=True)


