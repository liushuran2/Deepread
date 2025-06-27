import sys
import pycuda.autoinit  # 自动初始化 CUDA 上下文
import pycuda.driver as cuda
from PIL import Image
import argparse
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import tifffile as tiff
import imageio
import pandas as pd
from utils.rigid_correction import*
from utils.utils import *
from utils import frame_utils
from utils.frame_utils import *
from utils.flow_viz import *
from utils.utils import InputPadder, forward_interpolate, gaussian_blur
from model.loss import calculate_gradient_loss, calculate_mse


from datetime import datetime
# import wandb


# 
def train_one_epoch(model, optimizer, criterion, scheduler, train_dataloader, scaler, epoch, args, start_time, writer=None):
    """
    Training logic for an epoch
    # model: RAFT model
    # optimizer: optimizer
    # criterion: loss function, default is sequence_loss
    # scheduler: learning rate scheduler
    # train_dataloader: training dataloader
    # scaler: AMP scaler
    # epoch: current epoch
    # args: arguments
    # start: start time
    # writer: tensorboard writer

    :param epoch: Integer, current training epoch.
    :return: A log that contains average loss and metric in this epoch.
    """
    device = next(model.parameters()).device
    
    # debug time block
    debug_time = False
    if debug_time:
        epoch_times = pd.DataFrame()
        torch.cuda.synchronize(device)  
        ### advanced time recording
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        lapsed_time_data = 0
        lapsed_time_forward = 0
        lapsed_time_backward = 0

    # configure mode
    model.train()       
    epoch_start_time = datetime.now()

    pred_loss_list = []
    pred_epe_list = []
    
    # iterate over dataloader
    for i, data_blob in enumerate(train_dataloader):

        # TODO add noise
        if debug_time:
            start.record()
            image, flow, valid = [data_to_gpu(x, device) for x in data_blob]
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            lapsed_time_data += start.elapsed_time(end)       
        else:
            # image1, image2, flow, valid = [data_to_gpu(x, device) for x in data_blob]
            # GT_image
            image, flow, valid = [data_to_gpu(x, device) for x in data_blob]
            # mask_image
            # image1, image2, flow, gt, mask, valid = [data_to_gpu(x, device) for x in data_blob]
        
        # write image
        # io.imsave(os.path.join('test', 'image_gt.tif'), image[0,0].detach().cpu().numpy().squeeze())
        # io.imsave(os.path.join('test', 'image_ori.tif'), image[0,3].detach().cpu().numpy().squeeze())
        # cv2.imwrite(os.path.join('test', 'image2.tif'), image[0,:,].detach().cpu().numpy().squeeze())
        # cv2.imwrite(os.path.join('test', 'gt.tiff'), gt[2].detach().cpu().numpy().squeeze())
        # image2 = image[0].permute(0, 2, 3, 1).cpu().numpy() # H x W x C
        # flow_gt = flow[0].detach().cpu().numpy().squeeze()
        # flow_gt = np.transpose(np.array(flow_gt), (0, 2, 3, 1))
        # image2_warped = image_warp(image2, -flow_gt) # out H x W x C frame 2
        # io.imsave(os.path.join('test', 'image_warp.tif'), image2_warped.squeeze())
        # image1_mask = image1 * mask
        # image1_mask = image1_mask[2].detach().cpu().numpy().squeeze()
        # cv2.imwrite(os.path.join('test', 'image1_mask.tiff'), image1_mask)

        # learning rate adjustment, if one need to adjust it during the scheduler
        norm_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
        

        if args.add_noise:
            stdv = np.random.uniform(0.0, 0.15) # only for 0-1 normalization
            # image1 = (image1 + stdv * torch.randn(*image1.shape).cuda())
            image2 = (image2 + stdv * torch.randn(*image2.shape).cuda())


        # forward
        if debug_time:
            start.record()
            flow_predictions = model(image, iters=args.iters)   
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            lapsed_time_forward += start.elapsed_time(end)
        else:
            image = image.contiguous()
            # get first frame
            template = image[:,0,:,:,:]
            batchsize, timepoints, c, h, w = image.shape
            template = template.unsqueeze(1).expand(-1, timepoints, -1, -1, -1).contiguous()

            # time to batch
            template = template.view(batchsize * timepoints, c, h, w)
            image = image.view(batchsize * timepoints, c, h, w)

            # image2_tmp = image.permute(0,2,3,1)
            # flow_gt = flow.view(batchsize * timepoints, 2, h, w).permute(0,2,3,1)
            # image2_warp = image_warp_tensor(image2_tmp, -flow_gt)
            # image2_warp = image2_warp.detach().cpu().numpy().squeeze()
            # io.imsave(os.path.join('test', 'image_flow_gt.tif'), image2_warp)
            # io.imsave(os.path.join('test', 'flow_gt.tif'), flow_to_image(flow_gt.detach().cpu().numpy()[3,:,:,:]))

            # predict flow
            flow_predictions, data_predictions = model(template, image, timepoints=timepoints, iters=args.iters)

            # image2_tmp = image.permute(0,2,3,1)
            # flow_gt = flow.view(batchsize * timepoints, 2, h, w).permute(0,2,3,1)
            # image2_warp = image_warp_tensor(image2_tmp, -flow_predictions[-1][0].permute(0,2,3,1))
            # image2_warp = image2_warp.detach().cpu().numpy().squeeze()
            # io.imsave(os.path.join('test', 'image_flow_pr.tif'), image2_warp)
            # io.imsave(os.path.join('test', 'flow_pr.tif'), flow_to_image(np.transpose(flow_predictions[-1].detach().cpu().numpy()[0,3,:,:,:], axes=(1,2,0))))
        
        # compute loss
        if debug_time:
            start.record()
            flow_loss, data_loss, metrics = criterion(flow_predictions, flow, data_predictions, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)      
            # clip the gradient          
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            lapsed_time_backward += start.elapsed_time(end)
        else:
            flow_loss, data_loss, metrics = criterion(flow_predictions, flow, data_predictions, valid, args.gamma)
            # mask_image
            # loss, metrics = criterion(flow_predictions, flow, valid, mask, args.gamma)

            loss = flow_loss * 0.5 + data_loss * 0.5

            # loss += frame_wise_loss*2
            # loss = frame_wise_loss
            # image2_tmp = image2.permute(0, 2, 3, 1)# H x W x C
            # flow_pr = flow_predictions[-1] # 2 x H x W
            # flow_pr = flow_pr.permute(0, 2, 3, 1) # H x W x 2
            # image2_warped = image_warp_tensor(image2_tmp, -flow_pr) # out H x W x C frame 2
            # image2_warped = image2_warped.permute(0, 3, 1, 2)
            # cv2.imwrite(os.path.join('test', 'image2_warp_tensor.tiff'), image2_warped[0].detach().cpu().numpy().squeeze())

            # 反向传播时：在求导时开启侦测
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)      
            # clip the gradient          
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 


        # backward and optimize    
        scaler.step(optimizer)
        scaler.update()

        # lr adjustment. modified by yz. 02272023. using built-in scheduler
        lr= optimizer.param_groups[0]['lr']
        
        
        # adjus the learning rate
        scheduler.step()
        
        # append the loss
        pred_loss_list.append(loss.item())
        pred_epe_list.append(metrics['epe'])

        # log
        if writer is not None:
            writer.add_scalar(f'Train/loss', loss, epoch+1)
            writer.add_scalar('Train/lr', lr, epoch+1)
            writer.add_scalar(f'Train/flow_loss', flow_loss, epoch+1)
            writer.add_scalar(f'Train/data_loss', data_loss, epoch+1)

        # print
        if i % args.print_freq == 0 or i == len(train_dataloader) - 1: # print training results

            print_str = '[{}] Epoch[{}/{}], Step [{}/{}], lr:{:.2e} metrics: epe {}, 1 {}, 3 {}, 5 {}, loss: {:.2e}'.format(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"), epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                metrics['epe'],metrics['1px'],metrics['3px'],metrics['5px'], loss.item())
            print(print_str, flush=True)
            
            # write training txt file
            with open('{}/log.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')

            # save the flow to local tensor board
            if writer is not None:
                global_step = epoch * len(train_dataloader) + i

                # visualize the flow, hsv

                # for spynet
                # flow_predictions = [flow_predictions]

                flow_up = flow_predictions[-1]
                flow_pre_img = [flow_to_image(flow_up[j,2].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow_up.shape[0])]
                flow_pre_img = np.stack(flow_pre_img, axis=0).astype(np.float16) / 255.0 # to float
                # visualize_batch_flow(flow_pre_img)
                flow_gt_img = [flow_to_image(flow[j,2].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow.shape[0])]
                flow_gt_img = np.stack(flow_gt_img, axis=0).astype(np.float16) / 255.0
                # visualize_batch_flow(flow_gt_img)
                writer.add_image('Train/gt', make_grid(torch.from_numpy(flow_gt_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                writer.add_image('Train/pre', make_grid(torch.from_numpy(flow_pre_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                # for iii, intermediate_output in enumerate(flow_predictions):
                #         flow_pre_img = [flow_to_image(intermediate_output[j].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(intermediate_output.shape[0])]
                #         flow_pre_img = np.stack(flow_pre_img, axis=0)   .astype(np.float16) / 255.0         
                #         # Use make_grid if intermediate_output is an image or batch of images
                #         writer.add_image(f'Train/iter_{iii}', make_grid(torch.from_numpy(flow_pre_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)

                # visualize the flow, no hsv version but difference
                flow_difference_x = [direction_plot_flow(flow_up[j,2].detach().cpu().numpy(), flow[j,2].detach().cpu().numpy())[0] for j in range(flow_up.shape[0])]
                # for debuging purpose
                # visualize_flow(flow_difference_x[0])
                flow_difference_x = np.stack(flow_difference_x, axis=0).astype(np.float16) # to float
                flow_difference_y = [direction_plot_flow(flow_up[j,2].detach().cpu().numpy(), flow[j,2].detach().cpu().numpy())[1] for j in range(flow_up.shape[0])]
                flow_difference_y = np.stack(flow_difference_y, axis=0).astype(np.float16) # to float
                writer.add_image('Train/flow_difference_x', make_grid(torch.from_numpy(flow_difference_x).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                writer.add_image('Train/flow_difference_y', make_grid(torch.from_numpy(flow_difference_y).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)

                # visualize warp
                # combined_vis = []
                # for j in range(flow_up.shape[0]):
                #     image2_tmp = image2[j].permute(1, 2, 0).cpu().numpy() # H x W x C
                #     image1_tmp = image1[j].cpu().numpy() # C x H x W
                #     flow_pr = flow_up[j].cpu().detach().numpy() # 2 x H x W
                #     flow_pr = np.transpose(np.array(flow_pr), (1, 2, 0)) # H x W x 2
                #     image2_warped = image_warp(image2_tmp, -flow_pr) # out H x W x C frame 2
                #     image2_warped = np.transpose(image2_warped, (2, 0, 1))

                #     # binarize the image for better visualization
                #     combined_vis.append(show_warped_changes(image1_tmp, image2_warped))
                #     # combined_vis.append(show_warped_changes(np.where(image1_tmp > 0.8, 1, 0), np.where(image2_warped > 0.8, 1, 0)))
                #     # for debuging purpose
                #     # visualize_flow(combined_vis[0])
                # combined_vis = np.stack(combined_vis, axis=0).astype(np.float16) # to float
                # writer.add_image('Train/warped_result', make_grid(torch.from_numpy(combined_vis).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)


    if debug_time:
        timing_data = {
        'Epoch': epoch + 1,  # Epoch numbering typically starts from 1
        'lapsed_time_data (ms)': lapsed_time_data,
        'lapsed_time_forward (ms)': lapsed_time_forward,
        'lapsed_time_backward (ms)': lapsed_time_backward, 
        }
        epoch_times = pd.concat([epoch_times, pd.DataFrame([timing_data])], ignore_index=True)
        epoch_times.to_csv(f'epoch_{epoch}_times.csv', index=False)

    # print the time
    epoch_end_time = datetime.now()
    print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
            (epoch_end_time - start_time).total_seconds() / (epoch + 1 - args.start_epoch) ))
    
    return pred_loss_list, pred_epe_list



@torch.no_grad()
def evaluate(model, criterion, full_dataloader, args, epoch, writer=None):
    # model: full model
    # full_dataloader: all the frames
    # local_rank: the rank of the current gpu
    # args: the arguments
    # data_property: the property of the data
    # dump_vis: whether to dump the visualizations
    # huffman_coding: whether to do huffman coding
    
    model.eval()
    device = next(model.parameters()).device
    pred_loss_list = []
    pred_epe_list = []
    pred_data_list = []
    # go over each frame
    for i, data_blob in enumerate(full_dataloader):
        # image1, image2, flow, valid = [data_to_gpu(x, device) for x in data_blob]
        # GT_image
        image, flow, valid = [data_to_gpu(x, device) for x in data_blob]

        image = image.contiguous()
        # get first frame
        template = image[:,0,:,:,:]
        batchsize, timepoints, c, h, w = image.shape
        template = template.unsqueeze(1).expand(-1, timepoints, -1, -1, -1).contiguous()

        # time to batch
        template = template.view(batchsize * timepoints, c, h, w)
        image = image.view(batchsize * timepoints, c, h, w)

        # predict flow
        flow_predictions, data_predictions = model(template, image, timepoints=timepoints, iters=args.iters)    

        #spynet
        # flow_predictions = [flow_predictions]

        flow_up = flow_predictions[-1] # get flow up
        
        # compute loss, note this is different from the evaluation function in original code. We prepare a 
        flow_loss, data_loss, metrics = criterion(flow_predictions, flow, data_predictions, valid, args.gamma)

        loss = flow_loss * 0.5 + data_loss * 0.5
    

        # append the loss
        pred_loss_list.append(loss.item())
        pred_epe_list.append(metrics['epe'])
        pred_data_list.append(data_loss.item())

        # log
        if writer is not None:
            writer.add_scalar(f'Eval/loss', loss, epoch+1)
            writer.add_scalar(f'Eval/flow_loss', flow_loss, epoch+1)
            writer.add_scalar(f'Eval/data_loss', data_loss, epoch+1)

        # print the log
        if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
            print_str = '[{}] Eval at Step [{}/{}] , metrics: epe {}, 1 {}, 3 {}, 5 {}, loss: {:.2e}'.format(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"), i+1, len(full_dataloader), 
                metrics['epe'],metrics['1px'],metrics['3px'],metrics['5px'], loss.item())
 
            print(print_str, flush=True)
            with open('{}/log.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')

            # visualize the flow. save the flow to local tensor board
            if writer is not None:
                global_step = epoch * len(full_dataloader) + i
                flow_up = flow_predictions[-1]
                flow_pre_img = [flow_to_image(flow_up[j,2].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow_up.shape[0])]
                flow_pre_img = np.stack(flow_pre_img, axis=0).astype(np.float16) / 255.0
                # visualize_batch_flow(flow_pre_img)
                flow_gt_img = [flow_to_image(flow[j,2].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow.shape[0])]
                flow_gt_img = np.stack(flow_gt_img, axis=0).astype(np.float16) / 255.0
                # visualize_batch_flow(flow_gt_img)
                writer.add_image('Eval/gt', make_grid(torch.from_numpy(flow_gt_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                writer.add_image('Eval/pre', make_grid(torch.from_numpy(flow_pre_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                # for iii, intermediate_output in enumerate(flow_predictions):
                #         flow_pre_img = [flow_to_image(intermediate_output[j].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(intermediate_output.shape[0])]
                #         flow_pre_img = np.stack(flow_pre_img, axis=0)    .astype(np.float16) / 255.0         
                #         # Use make_grid if intermediate_output is an image or batch of images
                #         writer.add_image(f'Eval/iter_{iii}', make_grid(torch.from_numpy(flow_pre_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                
    return pred_loss_list, pred_epe_list, pred_data_list

@torch.no_grad()
# Done: the saving of output
# TODO how a sequential data is stored need to be rectified
def test_batch(model, args, data, session_name, data_property, video_template, batch_size=12, overlap_size=4, iters=12, warm_start=False, output_path='test'):
    """ Create test tiff file for input"""
    model.eval()
    device = next(model.parameters()).device
    flow_prev, sequence_prev = None, None
    (nframes, Lx, Ly) = data.shape

    frame_list = []
    flow_list = []

    steps = (nframes - 2*overlap_size) // batch_size

    # go over frames
    for i in range(steps):
        image = data[i * batch_size : (i + 1) * batch_size + 2*overlap_size].unsqueeze(0).unsqueeze(2)
        template = video_template.unsqueeze(0).unsqueeze(0)

        batchsize, timepoints, c, h, w = image.shape
        template = template.unsqueeze(1).expand(-1, timepoints, -1, -1, -1).contiguous()

        # time to batch
        template = template.view(batchsize * timepoints, c, h, w)
        image = image.view(batchsize * timepoints, c, h, w)

        # rigid correction
        for t in range(timepoints):
            img2rigid = image[t,0].detach().cpu().numpy()
            shifts, src_freq, phasediff = register_translation(img2rigid, template[0,0].detach().cpu().numpy(), 10)
            img_rigid = apply_shift_iteration(img2rigid, (-shifts[0], -shifts[1]))
            image[t,0] = torch.tensor(img_rigid, device=device)

        flow_pr, data_pr = model(template, image, iters=iters, timepoints=timepoints, flow_init=flow_prev, test_mode=True,denoise=True) # note flow_prev is only used for warm start

        # doublestage
        if args.doublestage:
            data_fisrt = data_pr.permute(0,2,1,3,4)
            template = torch.median(data_fisrt, dim=1, keepdim=True)[0].expand(-1, timepoints, -1, -1, -1).contiguous()
            template = template.view(batchsize * timepoints, c, h, w)
            data_fisrt = data_fisrt.view(batchsize * timepoints, c, h, w)
            flow_pr, data_pr = model(template, data_fisrt, iters=iters, timepoints=timepoints, flow_init=flow_prev, test_mode=True, denoise=False) # note flow_prev is only used for warm start
        if i == 0:
            data_pr_middle = data_pr[0,0, :batch_size+overlap_size].detach().cpu().numpy()
        elif i == steps - 1:
            data_pr_middle = data_pr[0,0, overlap_size:].detach().cpu().numpy()
        else:
            data_pr_middle = data_pr[0,0, overlap_size:batch_size+overlap_size].detach().cpu().numpy()
        data_pr_middle = adjust_frame_intensity(data_pr_middle)
        frame_list.extend(data_pr_middle)

        # 清理显存缓存
        torch.cuda.empty_cache()
        del flow_pr, data_pr

    # write optical flow to file    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # normalize the output
    video_array = np.stack(frame_list, axis=0).squeeze().astype(np.float32)
    
    # nan handle
    # video_array = np.nan_to_num(video_array, nan=0)
    # denorm_fun = lambda normalized_video: postprocessing_video(normalized_video, args.norm_type, data_property) 
    # video_array = denorm_fun(video_array)
    return video_array

def test_batch_tensorrt(engine, args, data, session_name, data_property, video_template, batch_size=12, overlap_size=4, iters=12, warm_start=False, output_path='test'):
    """ Create test tiff file for input"""
    # model.eval()
    # device = next(model.parameters()).device
    context = engine.create_execution_context()
    flow_prev, sequence_prev = None, None
    (nframes, Lx, Ly) = data.shape

    frame_list = []
    # flow_list = []

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
        if args.doublestage:
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

    # write optical flow to file    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # normalize the output
    video_array = np.stack(frame_list, axis=0).squeeze().astype(np.float32)

    return video_array

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

#For debug
def valid(model, criterion, full_dataloader, args):
    # model: full model
    # full_dataloader: all the frames
    # local_rank: the rank of the current gpu
    # args: the arguments
    # data_property: the property of the data
    # dump_vis: whether to dump the visualizations
    # huffman_coding: whether to do huffman coding
    
    model.eval()
    device = next(model.parameters()).device
    pred_loss_list = []
    pred_epe_list = []
    # go over each frame
    for i, data_blob in enumerate(full_dataloader):
        image1, image2, flow, valid = [data_to_gpu(x, device) for x in data_blob]

        # blur
        # image1_blur = gaussian_blur(image1)
        # image2_blur = gaussian_blur(image2)

        # forward
        flow_predictions = model(image1, image2, iters=12, test_mode=False)

        # blur
        # flow_predictions = model(image1_blur, image2_blur, iters=12, test_mode=False)   

        #spynet
        # flow_predictions = [flow_predictions]

        flow_up = flow_predictions[-1] # get flow up


        # compute loss, note this is different from the evaluation function in original code. We prepare a 
        loss, metrics = criterion(flow_predictions, flow, valid, 0.8)

        # append the loss
        pred_loss_list.append(loss.item())
        pred_epe_list.append(metrics['epe'])

        flow_up = flow_predictions[-1]
        flow_pre_img = [flow_to_image(flow_up[j].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow_up.shape[0])]
        flow_pre_img = np.stack(flow_pre_img, axis=0)
        cv2.imwrite(os.path.join('test', 'flow_pre', 'flow_pre_{}.tiff'.format(i)), flow_pre_img.squeeze())
        # visualize_batch_flow(flow_pre_img)
        flow_gt_img = [flow_to_image(flow[j].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow.shape[0])]
        flow_gt_img = np.stack(flow_gt_img, axis=0)
        cv2.imwrite(os.path.join('test', 'flow_gt', 'flow_gt_{}.tiff'.format(i)), flow_gt_img.squeeze())
        # visualize_batch_flow(flow_gt_img)

        # cv2.imwrite(os.path.join('test', 'image2.tiff'), image2.detach().cpu().numpy().squeeze())
        # cv2.imwrite(os.path.join('test', 'image1.tiff'), image1.detach().cpu().numpy().squeeze())
        image2 = image2[0].permute(1, 2, 0).cpu().numpy() # H x W x C
        flow_gt = flow.detach().cpu().numpy().squeeze()
        flow_pr = flow_up.detach().cpu().numpy().squeeze()
        # flow_gt = args.gt_flow[i]
        flow_gt = np.transpose(np.array(flow_gt), (1, 2, 0))
        flow_pr = np.transpose(np.array(flow_pr), (1, 2, 0))
        image2_warped = image_warp(image2, -flow_pr) # out H x W x C frame 2
        cv2.imwrite(os.path.join('test', 'data1', 'image2_warp_{}.tiff'.format(i)), image2_warped)

       
    return pred_loss_list, pred_epe_list
    # return 0

def finetune_one_epoch(model, optimizer, criterion, scheduler, train_dataloader, scaler, epoch, args, start_time, writer=None):
    """
    Training logic for an epoch
    # model: RAFT model
    # optimizer: optimizer
    # criterion: loss function, default is sequence_loss
    # scheduler: learning rate scheduler
    # train_dataloader: training dataloader
    # scaler: AMP scaler
    # epoch: current epoch
    # args: arguments
    # start: start time
    # writer: tensorboard writer

    :param epoch: Integer, current training epoch.
    :return: A log that contains average loss and metric in this epoch.
    """
    device = next(model.parameters()).device
    
    # debug time block
    debug_time = False
    if debug_time:
        epoch_times = pd.DataFrame()
        torch.cuda.synchronize(device)  
        ### advanced time recording
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        lapsed_time_data = 0
        lapsed_time_forward = 0
        lapsed_time_backward = 0

    # configure mode
    model.train()       
    epoch_start_time = datetime.now()

    pred_loss_list = []
    pred_epe_list = []
    
    # iterate over dataloader
    for i, data_blob in enumerate(train_dataloader):

        # TODO add noise
        if debug_time:
            start.record()
            image, flow, valid = [data_to_gpu(x, device) for x in data_blob]
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            lapsed_time_data += start.elapsed_time(end)       
        else:
            # image1, image2, flow, valid = [data_to_gpu(x, device) for x in data_blob]
            # GT_image
            image = data_to_gpu(data_blob, device)
            # mask_image
            # image1, image2, flow, gt, mask, valid = [data_to_gpu(x, device) for x in data_blob]
        
        # write image
        # io.imsave(os.path.join('test', 'image_gt.tif'), image[0,0].detach().cpu().numpy().squeeze())
        # io.imsave(os.path.join('test', 'image_ori.tif'), image[0,3].detach().cpu().numpy().squeeze())
        # cv2.imwrite(os.path.join('test', 'image2.tif'), image[0,:,].detach().cpu().numpy().squeeze())
        # cv2.imwrite(os.path.join('test', 'gt.tiff'), gt[2].detach().cpu().numpy().squeeze())
        # image2 = image[0].permute(0, 2, 3, 1).cpu().numpy() # H x W x C
        # flow_gt = flow[0].detach().cpu().numpy().squeeze()
        # flow_gt = np.transpose(np.array(flow_gt), (0, 2, 3, 1))
        # image2_warped = image_warp(image2, -flow_gt) # out H x W x C frame 2
        # io.imsave(os.path.join('test', 'image_warp.tif'), image2_warped.squeeze())
        # image1_mask = image1 * mask
        # image1_mask = image1_mask[2].detach().cpu().numpy().squeeze()
        # cv2.imwrite(os.path.join('test', 'image1_mask.tiff'), image1_mask)

        # learning rate adjustment, if one need to adjust it during the scheduler
        norm_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
        

        if args.add_noise:
            stdv = np.random.uniform(0.0, 0.15) # only for 0-1 normalization
            # image1 = (image1 + stdv * torch.randn(*image1.shape).cuda())
            image2 = (image2 + stdv * torch.randn(*image2.shape).cuda())


        # forward
        if debug_time:
            start.record()
            flow_predictions = model(image, iters=args.iters)   
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            lapsed_time_forward += start.elapsed_time(end)
        else:
            image = image.contiguous()
            # get first frame
            template = image[:,0,:,:,:]
            batchsize, timepoints, c, h, w = image.shape
            template = template.unsqueeze(1).expand(-1, timepoints, -1, -1, -1).contiguous()

            # time to batch
            template = template.view(batchsize * timepoints, c, h, w)
            image = image.view(batchsize * timepoints, c, h, w)

            # image2_tmp = image.permute(0,2,3,1)
            # flow_gt = flow.view(batchsize * timepoints, 2, h, w).permute(0,2,3,1)
            # image2_warp = image_warp_tensor(image2_tmp, -flow_gt)
            # image2_warp = image2_warp.detach().cpu().numpy().squeeze()
            # io.imsave(os.path.join('test', 'image_flow_gt.tif'), image2_warp)
            # io.imsave(os.path.join('test', 'flow_gt.tif'), flow_to_image(flow_gt.detach().cpu().numpy()[3,:,:,:]))

            # predict flow
            flow_predictions, data_predictions = model(template, image, timepoints=timepoints, iters=args.iters)

            # image2_tmp = image.permute(0,2,3,1)
            # flow_gt = flow.view(batchsize * timepoints, 2, h, w).permute(0,2,3,1)
            # image2_warp = image_warp_tensor(image2_tmp, -flow_predictions[-1][0].permute(0,2,3,1))
            # image2_warp = image2_warp.detach().cpu().numpy().squeeze()
            # io.imsave(os.path.join('test', 'image_flow_pr.tif'), image2_warp)
            # io.imsave(os.path.join('test', 'flow_pr.tif'), flow_to_image(np.transpose(flow_predictions[-1].detach().cpu().numpy()[0,3,:,:,:], axes=(1,2,0))))
        
        # compute loss
        if debug_time:
            start.record()
            flow_loss, data_loss, metrics = criterion(flow_predictions, flow, data_predictions, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)      
            # clip the gradient          
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            end.record()
            torch.cuda.synchronize()  # wait for all_reduce to complete
            lapsed_time_backward += start.elapsed_time(end)
        else:
            data_loss = criterion(data_predictions, args.gamma)
            # mask_image
            # loss, metrics = criterion(flow_predictions, flow, valid, mask, args.gamma)

            loss = data_loss

            # loss += frame_wise_loss*2
            # loss = frame_wise_loss
            # image2_tmp = image2.permute(0, 2, 3, 1)# H x W x C
            # flow_pr = flow_predictions[-1] # 2 x H x W
            # flow_pr = flow_pr.permute(0, 2, 3, 1) # H x W x 2
            # image2_warped = image_warp_tensor(image2_tmp, -flow_pr) # out H x W x C frame 2
            # image2_warped = image2_warped.permute(0, 3, 1, 2)
            # cv2.imwrite(os.path.join('test', 'image2_warp_tensor.tiff'), image2_warped[0].detach().cpu().numpy().squeeze())

            # 反向传播时：在求导时开启侦测
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)      
            # clip the gradient          
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 


        # backward and optimize    
        scaler.step(optimizer)
        scaler.update()

        # lr adjustment. modified by yz. 02272023. using built-in scheduler
        lr= optimizer.param_groups[0]['lr']
        
        # adjus the learning rate
        scheduler.step()
        
        # append the loss
        pred_loss_list.append(loss.item())
        # pred_epe_list.append(metrics['epe'])

        # log
        if writer is not None:
            writer.add_scalar(f'Train/loss', loss, epoch+1)
            writer.add_scalar('Train/lr', lr, epoch+1)
            # writer.add_scalar(f'Train/flow_loss', flow_loss, epoch+1)
            writer.add_scalar(f'Train/data_loss', data_loss, epoch+1)

        # print
        if i % args.print_freq == 0 or i == len(train_dataloader) - 1: # print training results

            print_str = '[{}] Epoch[{}/{}], Step [{}/{}], lr:{:.2e}  loss: {:.2e}'.format(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"), epoch+1, args.epochs, i+1, len(train_dataloader), lr, loss.item())
            print(print_str, flush=True)
            
            # write training txt file
            with open('{}/log.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')

            # save the flow to local tensor board
            if writer is not None:
                global_step = epoch * len(train_dataloader) + i

                # visualize the flow, hsv

                # for spynet
                # flow_predictions = [flow_predictions]

                # flow_up = flow_predictions[-1]
                # flow_pre_img = [flow_to_image(flow_up[j,2].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow_up.shape[0])]
                # flow_pre_img = np.stack(flow_pre_img, axis=0).astype(np.float16) / 255.0 # to float
                # # visualize_batch_flow(flow_pre_img)
                # flow_gt_img = [flow_to_image(flow[j,2].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow.shape[0])]
                # flow_gt_img = np.stack(flow_gt_img, axis=0).astype(np.float16) / 255.0
                # # visualize_batch_flow(flow_gt_img)
                # writer.add_image('Train/gt', make_grid(torch.from_numpy(flow_gt_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                # writer.add_image('Train/pre', make_grid(torch.from_numpy(flow_pre_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                # # for iii, intermediate_output in enumerate(flow_predictions):
                # #         flow_pre_img = [flow_to_image(intermediate_output[j].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(intermediate_output.shape[0])]
                # #         flow_pre_img = np.stack(flow_pre_img, axis=0)   .astype(np.float16) / 255.0         
                # #         # Use make_grid if intermediate_output is an image or batch of images
                # #         writer.add_image(f'Train/iter_{iii}', make_grid(torch.from_numpy(flow_pre_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)

                # # visualize the flow, no hsv version but difference
                # flow_difference_x = [direction_plot_flow(flow_up[j,2].detach().cpu().numpy(), flow[j,2].detach().cpu().numpy())[0] for j in range(flow_up.shape[0])]
                # # for debuging purpose
                # # visualize_flow(flow_difference_x[0])
                # flow_difference_x = np.stack(flow_difference_x, axis=0).astype(np.float16) # to float
                # flow_difference_y = [direction_plot_flow(flow_up[j,2].detach().cpu().numpy(), flow[j,2].detach().cpu().numpy())[1] for j in range(flow_up.shape[0])]
                # flow_difference_y = np.stack(flow_difference_y, axis=0).astype(np.float16) # to float
                # writer.add_image('Train/flow_difference_x', make_grid(torch.from_numpy(flow_difference_x).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                # writer.add_image('Train/flow_difference_y', make_grid(torch.from_numpy(flow_difference_y).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)

                # visualize warp
                # combined_vis = []
                # for j in range(flow_up.shape[0]):
                #     image2_tmp = image2[j].permute(1, 2, 0).cpu().numpy() # H x W x C
                #     image1_tmp = image1[j].cpu().numpy() # C x H x W
                #     flow_pr = flow_up[j].cpu().detach().numpy() # 2 x H x W
                #     flow_pr = np.transpose(np.array(flow_pr), (1, 2, 0)) # H x W x 2
                #     image2_warped = image_warp(image2_tmp, -flow_pr) # out H x W x C frame 2
                #     image2_warped = np.transpose(image2_warped, (2, 0, 1))

                #     # binarize the image for better visualization
                #     combined_vis.append(show_warped_changes(image1_tmp, image2_warped))
                #     # combined_vis.append(show_warped_changes(np.where(image1_tmp > 0.8, 1, 0), np.where(image2_warped > 0.8, 1, 0)))
                #     # for debuging purpose
                #     # visualize_flow(combined_vis[0])
                # combined_vis = np.stack(combined_vis, axis=0).astype(np.float16) # to float
                # writer.add_image('Train/warped_result', make_grid(torch.from_numpy(combined_vis).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)


    if debug_time:
        timing_data = {
        'Epoch': epoch + 1,  # Epoch numbering typically starts from 1
        'lapsed_time_data (ms)': lapsed_time_data,
        'lapsed_time_forward (ms)': lapsed_time_forward,
        'lapsed_time_backward (ms)': lapsed_time_backward, 
        }
        epoch_times = pd.concat([epoch_times, pd.DataFrame([timing_data])], ignore_index=True)
        epoch_times.to_csv(f'epoch_{epoch}_times.csv', index=False)

    # print the time
    epoch_end_time = datetime.now()
    print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
            (epoch_end_time - start_time).total_seconds() / (epoch + 1 - args.start_epoch) ))
    
    return pred_loss_list, pred_epe_list

@torch.no_grad()
def finetune_evaluate(model, criterion, full_dataloader, args, epoch, writer=None):
    # model: full model
    # full_dataloader: all the frames
    # local_rank: the rank of the current gpu
    # args: the arguments
    # data_property: the property of the data
    # dump_vis: whether to dump the visualizations
    # huffman_coding: whether to do huffman coding
    
    model.eval()
    device = next(model.parameters()).device
    pred_data_list = []
    # go over each frame
    for i, data_blob in enumerate(full_dataloader):
        # image1, image2, flow, valid = [data_to_gpu(x, device) for x in data_blob]
        # GT_image
        image = data_to_gpu(data_blob, device)

        image = image.contiguous()
        # get first frame
        template = image[:,0,:,:,:]
        batchsize, timepoints, c, h, w = image.shape
        template = template.unsqueeze(1).expand(-1, timepoints, -1, -1, -1).contiguous()

        # time to batch
        template = template.view(batchsize * timepoints, c, h, w)
        image = image.view(batchsize * timepoints, c, h, w)

        # predict flow
        flow_predictions, data_predictions = model(template, image, timepoints=timepoints, iters=args.iters)    

        #spynet
        # flow_predictions = [flow_predictions]

        flow_up = flow_predictions[-1] # get flow up
        
        # compute loss, note this is different from the evaluation function in original code. We prepare a 
        data_loss = criterion(data_predictions, args.gamma)

        loss = data_loss

        # append the loss
        pred_data_list.append(data_loss.item())

        # log
        if writer is not None:
            writer.add_scalar(f'Eval/loss', loss, epoch+1)
            writer.add_scalar(f'Eval/data_loss', data_loss, epoch+1)

        # print the log
        if i % args.print_freq == 0 or i == len(full_dataloader) - 1:
            print_str = '[{}] Eval at Step [{}/{}] , loss: {:.2e}'.format(
                datetime.now().strftime("%Y/%m/%d %H:%M:%S"), i+1, len(full_dataloader), loss.item())
 
            print(print_str, flush=True)
            with open('{}/log.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')

            # # visualize the flow. save the flow to local tensor board
            # if writer is not None:
            #     global_step = epoch * len(full_dataloader) + i
            #     flow_up = flow_predictions[-1]
            #     flow_pre_img = [flow_to_image(flow_up[j,2].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow_up.shape[0])]
            #     flow_pre_img = np.stack(flow_pre_img, axis=0).astype(np.float16) / 255.0
            #     # visualize_batch_flow(flow_pre_img)
            #     flow_gt_img = [flow_to_image(flow[j,2].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(flow.shape[0])]
            #     flow_gt_img = np.stack(flow_gt_img, axis=0).astype(np.float16) / 255.0
            #     # visualize_batch_flow(flow_gt_img)
            #     writer.add_image('Eval/gt', make_grid(torch.from_numpy(flow_gt_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
            #     writer.add_image('Eval/pre', make_grid(torch.from_numpy(flow_pre_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
            #     # for iii, intermediate_output in enumerate(flow_predictions):
            #     #         flow_pre_img = [flow_to_image(intermediate_output[j].detach().cpu().numpy().transpose(1, 2, 0)) for j in range(intermediate_output.shape[0])]
            #     #         flow_pre_img = np.stack(flow_pre_img, axis=0)    .astype(np.float16) / 255.0         
            #     #         # Use make_grid if intermediate_output is an image or batch of images
            #     #         writer.add_image(f'Eval/iter_{iii}', make_grid(torch.from_numpy(flow_pre_img).permute(0, 3, 1, 2), nrow=8, normalize=True), global_step)
                
    return pred_data_list


