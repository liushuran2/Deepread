
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
from suns.Online.functions_init import init_online, plan_fft2, init_online_Autothreshold
from suns.Online.functions_online import merge_2, merge_2_nocons, merge_complete, select_cons, \
    preprocess_online_batch, CNN_online_batch, separate_neuron_online_batch, separate_neuron_online_batch_Autothreshold, refine_seperate_cons_online, final_merge, extrace_trace
from suns.PostProcessing.combine import segs_results, unique_neurons2_simp, \
    group_neurons, piece_neurons_IOU, piece_neurons_consume, convert_mask
from scipy.io import savemat, loadmat
from suns.PreProcessing.preprocessing_functions import preprocess_video, \
    plan_fft, plan_mask2, load_wisdom_txt, export_wisdom_txt, \
    SNR_normalization, median_normalization, median_calculation, find_dataset, preprocess_complete
from tensorRT.loadengine import load_engine
    


# logging related
import wandb 
from wandb import sdk as wanbd_sdk
import socket
from datetime import datetime, timedelta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # test file
    parser.add_argument('--model_path', default='/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_scale_10_stack_28_50mW_fton10mW/', help='path to the trained model')
    # parser.add_argument('--suns_model_path', default='/mnt/nas01/LSR/DATA/2p_bench/suns/suns_h5/0323/Weights/model_best4220.pth', help='path to the trained model')
    parser.add_argument('--suns_model_path', default='/mnt/nas01/LSR/DATA/2p_bench/suns/0629/Weights/model_latest.pth', help='path to the trained model')
    parser.add_argument('--gt_flow', type=str, nargs='+', default=None, 
                        help='test file for evaluation')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--add_blur', default=False, action='store_true')
    parser.add_argument('--doublestage', default=True, action='store_true')
    parser.add_argument('--update_baseline', default=True, action='store_true')
    parser.add_argument('--frames_init', type=int, default=100, help='buffer size of initialization')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of denoising')
    parser.add_argument('--overlap_size', type=int, default=2, help='batch size of denoising')
    parser.add_argument('--max_neuron_num', type=int, default=1500, help='batch size of denoising')

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

    # load the network
    checkpoint_path = os.path.join(outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = nn.DataParallel(RAFT(args_model))
        # model = RAFT(args_model)
        # model = SPyNet('https://download.openmmlab.com/mmediting/restorers/''basicvsr/spynet_20210409-c6c1bd09.pth')
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.eval()

    trt_path = '/mnt/nas01/LSR/DATA/checkpt/RAFTCAD_result_multiscale_scale_10_stack_16_50mW/DeepIE_tensorRT.trt'
    engine = load_engine(trt_path)
    if engine is None:
        print("加载引擎失败。")

    context = engine.create_execution_context()
    
    # load SUNs CNN model
    # fff = get_shallow_unet()
    filename_CNN = args_eval.suns_model_path
    fff = nn.DataParallel(ShallowUNet(), device_ids=[0])
    fff.cuda()
    fff.eval
    checkpoint = torch.load(filename_CNN, map_location='cpu')
    fff.load_state_dict(checkpoint['state_dict'])

    frames_init = args_eval.frames_init

    # get the evaluating dataset
    test_dataloader_array = []
    args_eval.data_property  = []
    args_eval.norm_type = args_model.norm_type
    update_baseline = args_eval.update_baseline

    Params_post={
                # minimum area of a neuron (unit: pixels).
                'minArea': 70, 
                # average area of a typical neuron (unit: pixels) 
                'avgArea': 180,
                # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
                'thresh_pmap': 170, 
                # values higher than "thresh_mask" times the maximum value of the mask are set to one.
                'thresh_mask': 0.3, 
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

    root_path = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_nodentrites/'
    # root_path = '/mnt/nas01/LSR/DATA/2p/'
    # get the list of directories in the root path
    # directories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    # directories = ['671162628', '679353932']
    # directories = ['501271265', '501704220', '524691284', '531006860',  '603516552', '604145810', '607040613', '669233895', '671162628', '679353932']
    # directories.sort()  # Sort directories by number

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
    directory_path = root_path + 'test_dataset/scale_10/50mW/'  # path to the data
    # directory_path = 'DATA/'
    loadframe_path = directory_path
    all_files = os.listdir(loadframe_path)

    tiff_files = [filename for filename in all_files if filename.endswith('.tiff')]
    tiff_files.sort()

    save_path = os.path.join(root_path, 'result', 'scale_10', '50mW', 'RAFTCAD_result_multiscale_scale_10_stack_16_50mW_170')
    p = mp.Pool()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for fnames in tiff_files:
            # fnames = 'Fsim_clean.tiff'
            session_name = fnames.split('/')[-1].split('.')[0]
            datapath = loadframe_path + fnames
            # datapath = '/mnt/nas01/LSR/DATA/NAOMi_dataset/depthrange_200_nodentrites_lighter/NA_0.80_Hz_30_D_0_pow_130/1/Fsim_30mW.tiff'
            video_raw = tiff.imread(datapath).astype(np.float32)
            p1_val = np.percentile(video_raw, 1)
            p99_val = np.percentile(video_raw, 99)
            args_eval.data_property = {
            'p1': p1_val,
            'p99': p99_val,
            }
            (nframes, Lx, Ly) = video_raw.shape
            video_raw = preprocessing_img(video_raw, args_model.norm_type)
            video_raw =  torch.from_numpy(video_raw.copy())

            dims = (Lx, Ly)
            # zero-pad the lateral dimensions to multiples of 8, suitable for CNN
            rowsnb = math.ceil(Lx/8)*8
            colsnb = math.ceil(Ly/8)*8
            dimsnb = (rowsnb, colsnb)

            med_frame2 = np.ones((rowsnb, colsnb, 2), dtype='float32')
            video_input = np.ones((frames_init, rowsnb, colsnb), dtype='float32')        
            pmaps_b_init = np.ones((frames_init, Lx, Ly), dtype='uint8')        
            frame_SNR = np.ones(dimsnb, dtype='float32')
            pmaps_b = np.ones(dims, dtype='uint8')

            # new param
            minArea = Params_post['minArea']
            avgArea = Params_post['avgArea']
            thresh_pmap = Params_post['thresh_pmap']
            thresh_mask = Params_post['thresh_mask']
            thresh_COM0 = Params_post['thresh_COM0']
            thresh_COM = Params_post['thresh_COM']
            thresh_IOU = Params_post['thresh_IOU']
            thresh_consume = Params_post['thresh_consume']
            cons = Params_post['cons']
            # thresh_pmap_float = (Params_post['thresh_pmap']+1.5)/256
            thresh_pmap_float = (thresh_pmap+1)/256 # for published version
            (mask2, bf, fft_object_b, fft_object_c) = (None, None, None, None)

            if args_eval.doublestage:
                iters = 1
            else:
                iters = 2

            batch_size = args_eval.batch_size
            overlap_size = args_eval.overlap_size

            # init stage
            bb = video_raw[:frames_init]
            # template = torch.median(bb, dim=0, keepdim=False)[0]
            template = torch.median(video_raw, dim=0, keepdim=False)[0]
            # template = video_raw[0]

            start_batch_MC = time.time()
            video_adjust = test_batch_tensorrt(engine, args_eval, bb, session_name, args_eval.data_property, template, batch_size,overlap_size,
                iters=iters, warm_start=False, output_path=save_path)
            video_adjust_copy = video_adjust.copy()
            end_batch_MC = time.time()
            batch_MC = end_batch_MC - start_batch_MC
            print('Total time of init MC: {:4f} s'.format(batch_MC))
            
            # output_file = os.path.join(save_path, '{}_reg.tiff'.format(session_name))
            # # save_image(video_array, str(data_property["data_type"]), output_file)
            # save_image(video_adjust, 'uint16', output_file)
            # denorm_fun = lambda normalized_video: postprocessing_video(normalized_video, args_eval.norm_type, args_eval.data_property) 
            # video_adjust_copy = denorm_fun(video_adjust_copy)

            # video = tiff.imread('/mnt/nas/YZ_personal_storage/Private/MC/2p_benchtop/2p_148d/trial_2p_1/mov.tiff')
            # nframes = video_raw.shape[0]
            # video = np.array(video)
            # bb_=np.zeros((frames_init, rowsnb, colsnb), dtype='float32')
            # bb_ [:, :Lx, :Ly]= video[:frames_init]
            
            start_batch_Seg = time.time()
            frames_init_real = video_adjust_copy.shape[0]
            med_frame3, segs_all, recent_frames = init_online(
            video_adjust_copy, dims, dimsnb, video_input, pmaps_b_init, fff, thresh_pmap_float, Params_post, \
            med_frame2, mask2, bf, fft_object_b, fft_object_c, \
            useSF=False, useTF=False, useSNR=False, med_subtract=False, \
            batch_size_init=100, useWT=False, p=p)

            tuple_temp = merge_complete(segs_all[:frames_init_real], dims, Params_post)
            end_batch_Seg = time.time()
            batch_Seg = end_batch_Seg - start_batch_Seg
            print('Total time of init Seg: {:4f} s'.format(batch_Seg))

            # extract the trace
            trace = np.zeros((max_neuron_num, nframes), dtype='float32')
            trace = extrace_trace(tuple_temp, video_adjust, trace, frame_start=0)

            (bf, fft_object_b, fft_object_c) = (None, None, None)
            # video_template = torch.median(torch.from_numpy(video_adjust.copy()), dim=0, keepdim=False)[0]
            # template = video_template.unsqueeze(0).unsqueeze(0)
            # video_template = torch.median(torch.from_numpy(video_adjust), dim=0, keepdim=False)[0]
            # bb=np.zeros(dimsnb, dtype='float32')
            template = template.unsqueeze(0).unsqueeze(0)
            whole_template = template


            print('Start batch by batch processing')

            # Online processing
            t_merge = frames_init_real
            frames_left = nframes - frames_init_real - overlap_size
            frame_list = []

            steps = (frames_left - 2*overlap_size) // batch_size

            data_buffer = queue.Queue(maxsize=frames_init)

            for i in range(frames_init_real):
                data_buffer.put(video_adjust_copy[i])               

            # go over frames
            for i in range(steps):
                frame_start = frames_init_real + i * batch_size
                image = video_raw[frames_init_real + i * batch_size : frames_init_real + (i + 1) * batch_size + 2*overlap_size].unsqueeze(0).unsqueeze(2)

                batchsize, timepoints, c, h, w = image.shape
                
                template = template.unsqueeze(1).expand(-1, timepoints, -1, -1, -1).contiguous()
                # template = template.view(batchsize * timepoints, c, h, w)

                # time to batch
                template = template.view(batchsize * timepoints, c, h, w)
                image = image.view(batchsize * timepoints, c, h, w)

                start_online_MC = time.time()
                # 分配缓冲区
                output_data = np.empty([1, 1, batch_size + 2 * overlap_size, 512, 512], dtype=np.float32)
                input_data = np.concatenate((image, template), axis=0)

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

                # doublestage
                data_fisrt = np.transpose(output_data, (0, 2, 1, 3, 4))
                # template = np.median(data_fisrt, axis=1, keepdims=False)
                # template = template.detach().cpu().numpy()
                data_fisrt = data_fisrt.reshape(batchsize * timepoints, c, h, w)

                # 分配缓冲区
                output_data = np.empty([1, 1, batch_size + 2 * overlap_size, 512, 512], dtype=np.float32)
                input_data = np.concatenate((image, template), axis=0)

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

                data_pr = output_data

                if i == 0:
                    data_pr_middle = data_pr[0,0, :batch_size + overlap_size]
                elif i == steps - 1:
                    data_pr_middle = data_pr[0,0, overlap_size:]
                else:
                    data_pr_middle = data_pr[0,0, overlap_size:overlap_size+batch_size]
                data_pr_middle = adjust_frame_intensity(data_pr_middle)
                frame_list.extend(data_pr_middle)

                end_online_MC = time.time()
                online_MC = end_online_MC - start_online_MC
                print('Batch time of online MC: {:4f} s'.format(online_MC))

                # update template
                # for j in range(data_pr_middle.shape[0]):
                #     if data_buffer.full():
                #         data_buffer.get()
                #     data_buffer.put(data_pr_middle[j])

                # data_buffer_list = list(data_buffer.queue)
                # new_template = np.median(np.stack(data_buffer_list, axis=0), axis=0)

                # 清理显存缓存                             
                torch.cuda.empty_cache()
                # del flow_pr, data_pr
                
                # new_template = np.median(data_pr_middle, axis=0)
                # old_template = template[0,0].detach().cpu().numpy()

                # new_template = 0.96 * old_template + 0.04 * new_template

                # template = torch.from_numpy(new_template).unsqueeze(0).unsqueeze(0)
                # template = torch.median(video_raw, dim=0, keepdim=False)[0].unsqueeze(0).unsqueeze(0)
                template = whole_template

                start_online_Seg = time.time()

                # preprocess
                frames_SNR = np.ones((data_pr_middle.shape[0], rowsnb, colsnb), dtype='float32')        
                # pmaps_b_init = np.ones((data_pr_middle.shape[0], Lx, Ly), dtype='uint8')  
                bb = data_pr_middle.copy()
                # frame_SNR = np.ones(dimsnb, dtype='float32')
                pmaps_b = np.ones((data_pr_middle.shape[0], rowsnb, colsnb), dtype='uint8')
                
                frames_SNR = preprocess_online_batch(bb, dimsnb, dimsnb, med_frame3, frame_SNR, \
                    None, None, None, None, \
                    None, useSF=False, useTF=False, useSNR=False, \
                    med_subtract=False, update_baseline=False)
                
                # import tifffile as tiff
                # tiff.imwrite('test/frames_SNR.tif', frames_SNR, dtype='float32')
                
                # CNN inference
                frames_prob = CNN_online_batch(frames_SNR[:, :rowsnb, :colsnb], fff, batch_size)
                # tiff.imwrite('test/frames_prob.tif', frames_prob, dtype='float32')
                
                # first step of post-processing
                segs = separate_neuron_online_batch(frames_prob, pmaps_b, thresh_pmap_float, minArea, avgArea, useWT=False, useMP=True, p=p)
                segs_all.extend(segs)

                totalmasks, neuronstate, COMs, areas, probmapID = segs_results(segs_all[t_merge:])
                uniques, times_uniques = unique_neurons2_simp(totalmasks, neuronstate, COMs, \
                    areas, probmapID, minArea=0, thresh_COM0=thresh_COM0, useMP=True)
                
                if uniques.size:
                    groupedneurons, times_groupedneurons = \
                        group_neurons(uniques, thresh_COM, thresh_mask, dims, times_uniques, useMP=True)
                    piecedneurons_1, times_piecedneurons_1 = \
                        piece_neurons_IOU(groupedneurons, thresh_mask, thresh_IOU, times_groupedneurons)
                    
                    piecedneurons, times_piecedneurons = \
                        piece_neurons_consume(piecedneurons_1, avgArea, thresh_mask, thresh_consume, times_piecedneurons_1)
                    # masks of new neurons
                    masks_add = piecedneurons
                    # indices of frames when the neurons are active
                    times_add = [np.unique(x) + t_merge for x in times_piecedneurons]
                        
                    # Refine neurons using consecutive occurence
                    if masks_add.size:
                        # new real-number masks
                        masks_add = [x for x in masks_add]
                        # new binary masks
                        Masksb_add = [(x >= x.max() * thresh_mask).astype('float') for x in masks_add]
                        # areas of new masks
                        area_add = np.array([x.nnz for x in Masksb_add])
                        # indicators of whether the new masks satisfy consecutive frame requirement
                        have_cons_add = refine_seperate_cons_online(times_add, cons)
                    else:
                        Masksb_add = []
                        area_add = np.array([])
                        have_cons_add = np.array([])
                else: # does not find any active neuron
                    Masksb_add = []
                    masks_add = []
                    times_add = times_uniques
                    area_add = np.array([])
                    have_cons_add = np.array([])
                tuple_add = (Masksb_add, masks_add, times_add, area_add, have_cons_add)

                tuple_temp = merge_2(tuple_temp, tuple_add, dims, Params_post)
                t_merge += batch_size

                end_online_Seg = time.time()
                online_Seg = end_online_Seg - start_online_Seg
                print('Batch time of online Seg: {:4f} s'.format(online_Seg))

                # extract the trace
                start_online_tr = time.time()
                trace = extrace_trace(tuple_temp, data_pr_middle, trace, frame_start=frame_start)
                end_online_tr = time.time()
                online_tr = end_online_tr - start_online_tr
                print('Batch time of online extracting trace: {:4f} s'.format(online_tr))

            # final merging
            Masks_2, times_active = final_merge(tuple_temp, Params_post)
            Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')

            # normalize the output
            video_array = np.stack(frame_list, axis=0).squeeze().astype(np.float32)
            video_array = np.concatenate((video_adjust, video_array), axis=0).astype(np.float32)
            
            # nan handle
            video_array = np.nan_to_num(video_array, nan=0)
            denorm_fun = lambda normalized_video: postprocessing_video(normalized_video, args_eval.norm_type, args_eval.data_property) 
            video_array = denorm_fun(video_array)

            # video_array = np.concatenate((video_adjust, video_array), axis=0).astype(np.float32)

            output_file = os.path.join(save_path, '{}_reg.tiff'.format(session_name))
            # save_image(video_array, str(data_property["data_type"]), output_file)
            save_image(video_array, 'uint16', output_file)

            savemat(os.path.join(save_path, '{}_Output_Masks.mat'.format(session_name)), \
        {'Masks':Masks, 'times_active':times_active}, do_compression=True)
            
            savemat(os.path.join(save_path, '{}_Extracted_trace.mat'.format(session_name)), \
        {'trace':trace}, do_compression=True)

            # ender.record()
            # curr_time = starter.elapsed_time(ender)
            # print(curr_time)


