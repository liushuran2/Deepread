
from __future__ import print_function, division
import sys

import argparse
import configparser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
import yaml

import torch
import torch.nn as nn
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
from model.loss import sequence_loss


# logging related
import wandb 
from wandb import sdk as wanbd_sdk
import socket
from datetime import datetime, timedelta

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # test file
    parser.add_argument('--model_path', default='checkpt/RAFTCAD_result_multiscale_stack_2002/', help='path to the trained model')
    parser.add_argument('--test_file', type=str, nargs='+', default=['/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_1.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_2.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_3.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_4.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_5.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_6.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_7.tiff', 
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_8.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_9.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_10.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_11.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_12.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_13.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_14.tiff',
                                                                     '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_15.tiff',],

                        help='test file for evaluation')
    parser.add_argument('--gt_flow', type=str, nargs='+', default=['/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/testlong/motion_N_800_scale_10_noise.mat'], 
                        help='test file for evaluation')
    parser.add_argument('-b', '--batchSize', type=int, default=8, help='input batch size')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--add_blur', default=False, action='store_true')
    parser.add_argument('--doublestage', default=False, action='store_true')

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
    checkpoint_path = os.path.join(args_model.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = nn.DataParallel(RAFT(args_model))
        # model = SPyNet('https://download.openmmlab.com/mmediting/restorers/''basicvsr/spynet_20210409-c6c1bd09.pth')
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.eval()

    # get the evaluating dataset
    test_dataloader_array = []
    args_eval.data_property  = []
    args_eval.norm_type = args_model.norm_type
    for datapath in  args_eval.test_file:
        test_dataset = FlowTestDataset(args_model, datapath)
        args_eval.data_property.append(test_dataset.data_property) # call the data_property for getting the data info
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, # note shuffle is disabled, and batchsize has to be 1
                                                        num_workers=args_model.workers, pin_memory=True, persistent_workers=True,
                                                        drop_last=False, worker_init_fn=worker_init_fn)
        test_dataloader_array.append(test_dataloader)

    # do file by file evaluating
    for i, (test_dataloader, test_file_name) in enumerate(zip(test_dataloader_array, args_eval.test_file)):
        session_name = test_file_name.split('/')[-1].split('.')[0]
        test(model, args_eval, test_dataloader, session_name, args_eval.data_property[i], iters=22, warm_start=False, output_path=outf)


    # TODO show the differences between the ground truth and the predicted flow

    # For debug, train data set
    # valid_dataset = FlowValidDataset('/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/specified_N_200_scale_10_noise_withGT.h5', None)
    # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1,  shuffle=False,
    #                                                num_workers=1, pin_memory=True, persistent_workers=True,
    #                                                drop_last=False, worker_init_fn=worker_init_fn)
    # criterion = sequence_loss
    # args_eval.gt_flow = scipy.io.loadmat('/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/motion_N_200_scale_10_noise.mat')['motion_field']
    # valid(model, criterion, valid_dataloader, args_eval)


