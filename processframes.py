
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
from utils.rigid_correction import * 
import pycuda.driver as cuda
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
from utils.frame_utils import *
from model.loss import sequence_loss
from suns.Network.shallow_unet import get_shallow_unet
from suns.Network.shallow_unet import ShallowUNet
from suns.Online.functions_init import init_online, plan_fft2
from suns.Online.functions_online import merge_2, merge_2_nocons, merge_complete, select_cons, \
    preprocess_online_batch, CNN_online_batch, separate_neuron_online_batch, refine_seperate_cons_online, final_merge
from suns.PostProcessing.combine import segs_results, unique_neurons2_simp, \
    group_neurons, piece_neurons_IOU, piece_neurons_consume, convert_mask
from scipy.io import savemat, loadmat
from suns.PreProcessing.preprocessing_functions import preprocess_video, \
    plan_fft, plan_mask2, load_wisdom_txt, export_wisdom_txt, \
    SNR_normalization, median_normalization, median_calculation, find_dataset, preprocess_complete
from tensorRT.loadengine import load_engine

def extrace_trace(Masks, prob_map, frame_start=0):
    N = len(Masks) # component number
    # print('Neuron number: ', N)
    nframes, Lx, Ly = prob_map.shape
    trace = np.zeros((N, nframes), dtype=np.float32)

    for i in range(N):
        # read the target positions
        mask = np.reshape(Masks[i], (Lx, Ly))
        mask_valid, roi = convert_mask(mask)
        curr_temporal_signal = np.squeeze(np.mean(prob_map[:,roi[0]:roi[1],roi[2]:roi[3]] * mask_valid.astype(np.float32), axis=(1,2))) # broadcast here.
        trace[i] = curr_temporal_signal

    return trace

def test_batch_tensorrt(engine, data, video_template, batch_size=12, overlap_size=4):
    """ Create test tiff file for input"""
    import pycuda.autoinit  # This is needed to initialize the PyCUDA driver
    context = engine.create_execution_context()
    (nframes, Lx, Ly) = data.shape

    frame_list = []
    steps = (nframes - 2*overlap_size) // batch_size

    # go over frames
    for i in range(steps):
        image = data[i * batch_size : (i + 1) * batch_size + 2*overlap_size].unsqueeze(0).unsqueeze(2)
        template = video_template.unsqueeze(0).unsqueeze(0)

        batchsize, timepoints, c, h, w = image.shape
        template = template.contiguous()

        # time to batch
        template = template.view(1, c, h, w).detach().cpu().numpy()
        image = image.view(batchsize * timepoints, c, h, w).detach().cpu().numpy()

        # rigid correction
        for t in range(timepoints):
            img2rigid = image[t,0]
            shifts, src_freq, phasediff = register_translation(img2rigid, template[0,0], 10)
            img_rigid = apply_shift_iteration(img2rigid, (-shifts[0], -shifts[1]))
            image[t,0] = img_rigid


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
            data_pr_middle = data_pr[0,0, :batch_size+overlap_size]
        elif i == steps - 1:
            data_pr_middle = data_pr[0,0, overlap_size:]
        else:
            data_pr_middle = data_pr[0,0, overlap_size:batch_size+overlap_size]
        data_pr_middle = adjust_frame_intensity(data_pr_middle)
        frame_list.extend(data_pr_middle)

        # 清理显存缓存
        torch.cuda.empty_cache()
        del data_pr

    # normalize the output
    video_array = np.stack(frame_list, axis=0).squeeze().astype(np.float32)

    return video_array

def process_frames_init(video_raw, Params_post, batch_size, overlap_size, engine, fff, p):
    video_raw = preprocessing_img(video_raw, 'robust')
    video_raw =  torch.from_numpy(video_raw.copy())
    template = torch.median(video_raw, dim=0, keepdim=False)[0]

    video_adjust = test_batch_tensorrt(engine, video_raw, template, batch_size, overlap_size)
    video_adjust_copy = video_adjust.copy()

    Masks = seg_batch(video_adjust_copy, fff, p, Params_post, 20)

    # extract traces
    traces = extrace_trace(Masks, video_adjust, frame_start=0)
    return video_adjust, template, Masks, traces

def seg_batch(video_adjust, fff, p, Params_post, batch_size, frames_init_seg=100):  
    (nframes, Lx, Ly) = video_adjust.shape
    dims = (Lx, Ly)
    # zero-pad the lateral dimensions to multiples of 8, suitable for CNN
    rowsnb = math.ceil(Lx/8)*8
    colsnb = math.ceil(Ly/8)*8
    dimsnb = (rowsnb, colsnb)
    minArea = Params_post['minArea']
    avgArea = Params_post['avgArea']
    thresh_pmap = Params_post['thresh_pmap']
    thresh_mask = Params_post['thresh_mask']
    thresh_COM0 = Params_post['thresh_COM0']
    thresh_COM = Params_post['thresh_COM']
    thresh_IOU = Params_post['thresh_IOU']
    thresh_consume = Params_post['thresh_consume']
    cons = Params_post['cons']

    med_frame2 = np.ones((rowsnb, colsnb, 2), dtype='float32')
    video_input = np.ones((frames_init_seg, rowsnb, colsnb), dtype='float32')        
    pmaps_b_init = np.ones((frames_init_seg, Lx, Ly), dtype='uint8')

    thresh_pmap = Params_post['thresh_pmap']
    # thresh_pmap_float = (Params_post['thresh_pmap']+1.5)/256
    thresh_pmap_float = (thresh_pmap+1)/256 # for published version
    t_merge = frames_init_seg

    med_frame3, segs_all, recent_frames = init_online(
                video_adjust[:t_merge], dims, dimsnb, video_input, pmaps_b_init, fff, thresh_pmap_float, Params_post, \
                med_frame2,\
                useSF=False, useTF=False, useSNR=False, med_subtract=False, \
                batch_size_init=100, useWT=False, p=p)
    
    tuple_temp = merge_complete(segs_all, dims, Params_post)

    steps = (nframes - frames_init_seg) // batch_size

    # go over frames
    for i in range(steps):
        data_pr_middle = video_adjust[frames_init_seg + i * batch_size : frames_init_seg + (i + 1) * batch_size]
        frames_SNR = np.ones((data_pr_middle.shape[0], rowsnb, colsnb), dtype='float32')        
        # pmaps_b_init = np.ones((data_pr_middle.shape[0], Lx, Ly), dtype='uint8')  
        bb = data_pr_middle.copy()
        # frame_SNR = np.ones(dimsnb, dtype='float32')
        pmaps_b = np.ones((data_pr_middle.shape[0], rowsnb, colsnb), dtype='uint8')
        frames_SNR = preprocess_online_batch(bb, dimsnb, dimsnb, med_frame3, frames_SNR, \
            None, None, None, None, \
            None, useSF=False, useTF=False, useSNR=False, \
            med_subtract=False, update_baseline=False)
        
        for i in range(frames_SNR.shape[0]):
            img = frames_SNR[i]
            min_val = np.percentile(img, 88)
            max_val = np.percentile(img, 99.8)
            if max_val > min_val:
                temp = (img - min_val) / (max_val - min_val)
                # clip
                frames_SNR[i] = np.clip(temp, 0, 1)
            else:
                frames_SNR[i] = img  # 如果全为常数，保持原样
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

    # final merging
    Masks_2, times_active = final_merge(tuple_temp, Params_post)
    Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')

    return Masks

def process_frames_online(video_raw, template, batch_size, overlap_size, context, mask, traces):
    image = video_raw.unsqueeze(0).unsqueeze(2)
    batchsize, timepoints, c, h, w = image.shape

    template = template.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(-1, timepoints, -1, -1, -1).contiguous()
    # time to batch
    template = template.view(batchsize * timepoints, c, h, w)
    image = image.view(batchsize * timepoints, c, h, w)

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
    data_pr_middle = data_pr[0,0, overlap_size:]
    torch.cuda.empty_cache()






