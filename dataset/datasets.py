# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import os
import math
import random
from glob import glob
import os.path as osp
import tifffile as tiff

from utils.frame_utils import *
from utils.flow_viz import *
from dataset.augmentor import FlowAugmentor, FinetuneAugmentor
from utils.utils import bin_median


# modified for 2p usage, training purpose
class FlowDataset(data.Dataset):
    def __init__(self, args, hdf5_file, aug_params=None):
        self.augmentor = None
        if aug_params is not None: # change to grayscale augmentation
            self.augmentor = FlowAugmentor(**aug_params) 

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.norm_type = args.norm_type

        # let's read the hdf5 file
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as file:
            self.length = file['image_pairs'].shape[-1]  # size of h x w x 2 x n
            
    # follow suggestion from https://github.com/pytorch/pytorch/issues/11929
    def open_hdf5(self):
        self.img_hdf5 = h5py.File(self.hdf5_file, 'r')
        self.img_list = self.img_hdf5['image_pairs'] # if you want dataset.   
        self.flow_list= self.img_hdf5['motions']
        self.img_list = np.transpose(self.img_list, axes=(3,2,1,0))
        self.flow_list = np.transpose(self.flow_list, axes=(4,3,2,1,0))
        # mask_image
        # self.mask_list = self.img_hdf5['mask']

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        index = index % self.length
        valid = None
              
        # open the hdf5 file
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
            
        # Read a pair of images
        img = self.img_list[index, :, :, :] # should be 8 x 512 x 512
        flow = self.flow_list[index, :, :, :, :] # should be 8 x 2 x 512 x 512

        # io.imsave(os.path.join('test', 'image_gt_d.tif'), img[0,:,:])
        # io.imsave(os.path.join('test', 'image_ori_d.tif'), img[3,:,:])
        # # # cv2.imwrite(os.path.join('test', 'image2.tif'), image[0,:,].detach().cpu().numpy().squeeze())
        # # # cv2.imwrite(os.path.join('test', 'gt.tiff'), gt[2].detach().cpu().numpy().squeeze())
        # image2_warped = image_warp(img[3,:,:], -np.transpose(flow[3,:,:,:], axes=(1,2,0))) # out H x W x C frame 2
        # io.imsave(os.path.join('test', 'image_warp_d.tif'), image2_warped)
        # cv2.imwrite('test/flow_d.tif', flow_to_image(np.transpose(flow[3,:,:,:], axes=(1,2,0))))

        # img1 = img[:, :, 0] # should be 512 x 512
        # img2 = img[:, :, 1] # should be 512 x 512

        # rotate the img and the flow
        # img = np.transpose(np.flip(img, axis=0), axes=(1,0,2))
        # flow = np.transpose(np.flip(flow, axis=0), axes=(1,0,2,3))

        # pre-processing
        img = preprocessing_img(img, self.norm_type)
        
        # Read the flow
        flow = np.array(flow).astype(np.float32)
        
        # cast to numpy
        # flow = np.array(flow).astype(np.float32)
        img = np.array(img).astype(np.float32)[:, np.newaxis,:,:]

        # Apply data augmentation
        if self.augmentor is not None:
            img, flow = self.augmentor(img, flow)

        # Convert to PyTorch tensors
        img = torch.from_numpy(img).float() # change to 8 x 1 x 512 x 512

        flow = torch.from_numpy(flow).float() # change to 8 x 2 x 512 x 512

        # debugging, copy and paste it to the debugger console
        # plot_images_and_flow(img1, img2, flow)

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000) # the valid function constrains the flow to be within 1000

        # return img1, img2, flow, valid.float()
        # GT_image
        return img, flow, valid.float()
        # mask_image
        # return img1, img2, flow, gt, mask, valid.float()

    def __del__(self):
        if hasattr(self, 'img_hdf5'):
            self.img_hdf5.close()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.gt_list = v * self.gt_list
        return self
        
    def __len__(self):
        return self.length


class FlowTestDataset(data.Dataset):
    def __init__(self, args, data_path):
        self.augmentor = None
        self.is_test = True
        self.init_seed = False
        self.image_list = []

        # read the tiff file
        self.video =tiff.imread(data_path) # thw
        data_type = self.video.dtype

        self.video = self.video.astype(np.float32)
        mean_val =np.mean(self.video) 
        max_val = np.max(self.video)
        min_val = np.min(self.video)
        std_val = np.std(self.video)
        p1_val = np.percentile(self.video, 1)
        p99_val = np.percentile(self.video, 99)
        # video information
        self.data_property = {
            'mean': mean_val,
            'max': max_val,
            'min': min_val,
            'std': std_val,
            'p1': p1_val,
            'p99': p99_val,
            'data_type': data_type,
            'norm_type': args.norm_type
        }

        #
        self.video = preprocessing_img(self.video, args.norm_type)
        # self.summary_image = bin_median(self.video)

        # convert to tensor
        self.video =  torch.from_numpy(self.video.copy()) # t1hw


        # get the summary image, depend on summary image type
        # self.summary_image = self.video.mean(0)
        self.summary_image = torch.median(self.video, dim=0, keepdim=False)[0]
        # self.summary_image = self.video[0]

    def __getitem__(self, index):

        # Read a pair of images
        img = self.video[index * 8 : index * 8 + 16] # should be 8 x 512 x 512
        

        # cast to numpy
        img = np.array(img).astype(np.float32)
 

        # Convert to PyTorch tensors     
        img = torch.from_numpy(img).unsqueeze(1)  # change to 8 x 1 x 512 x 512
        return img, self.summary_image


    def __len__(self):
        return (len(self.video) - 8) // 8
    

class FlowValidDataset(data.Dataset):
    def __init__(self, hdf5_file, aug_params=None):
        self.augmentor = None
        if aug_params is not None: # change to grayscale augmentation
            self.augmentor = FlowAugmentor(**aug_params) 

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.norm_type = 'robust'

        # let's read the hdf5 file
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as file:
            self.length = file['image_pairs'].shape[-1]  # size of h x w x 2 x n
            
    # follow suggestion from https://github.com/pytorch/pytorch/issues/11929
    def open_hdf5(self):
        self.img_hdf5 = h5py.File(self.hdf5_file, 'r')
        self.img_list = self.img_hdf5['image_pairs'] # if you want dataset.   
        self.flow_list= self.img_hdf5['motions']

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        index = index % self.length
        valid = None
              
        # open the hdf5 file
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
            
        # Read a pair of images
        img = self.img_list[:, :, :, index] # should be 512 x 512 x 2
        flow = self.flow_list[:, :, :, index] # should be 512 x 512 x 2

        # pre-processing
        img = preprocessing_img(img, self.norm_type)

        img1 = np.rot90(np.flip(img[:, :, 0], axis=0), k=-1) # should be 512 x 512
        img2 = np.rot90(np.flip(img[:, :, 1], axis=0), k=-1) # should be 512 x 512
        flow1 = np.rot90(np.flip(flow[:, :, 0], axis=0), k=-1) # should be 512 x 512
        flow2 = np.rot90(np.flip(flow[:, :, 1], axis=0), k=-1) # should be 512 x 512

        # cv2.imwrite(os.path.join('test', 'image2.tiff'), img2.squeeze())
        # cv2.imwrite(os.path.join('test', 'image1.tiff'), img1.squeeze())
        # Read the flow
        
        # cast to numpy
        flow = np.stack([flow1, flow2], axis=2)
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.float32)[..., np.newaxis]
        img2 = np.array(img2).astype(np.float32)[..., np.newaxis]

        # Apply data augmentation
        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        # Convert to PyTorch tensors
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() # change to 1 x 512 x 512
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() # change to 1 x 512 x 512
        flow = torch.from_numpy(flow).permute(2, 0, 1).float() # change to 2 x 512 x 512

        # debugging, copy and paste it to the debugger console
        # plot_images_and_flow(img1, img2, flow)

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000) # the valid function constrains the flow to be within 1000

        return img1, img2, flow, valid.float()

    def __del__(self):
        if hasattr(self, 'img_hdf5'):
            self.img_hdf5.close()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return self.length
    
# modified for 2p usage, training purpose
class FinetuneDataset(data.Dataset):
    def __init__(self, args, hdf5_file, aug_params=None):
        self.augmentor = None
        if aug_params is not None: # change to grayscale augmentation
            self.augmentor = FinetuneAugmentor(**aug_params) 

        self.is_test = False
        self.init_seed = False
        self.image_list = []
        self.norm_type = args.norm_type

        # let's read the hdf5 file
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as file:
            self.length = file['image_pairs'].shape[-1]  # size of h x w x 2 x n
            
    # follow suggestion from https://github.com/pytorch/pytorch/issues/11929
    def open_hdf5(self):
        self.img_hdf5 = h5py.File(self.hdf5_file, 'r')
        self.img_list = self.img_hdf5['image_pairs'] # if you want dataset.   
        self.img_list = np.transpose(self.img_list, axes=(3,2,1,0))
        # mask_image
        # self.mask_list = self.img_hdf5['mask']

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        index = index % self.length
        valid = None
              
        # open the hdf5 file
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
            
        # Read a pair of images
        img = self.img_list[index, :, :, :] # should be 8 x 512 x 512

        # io.imsave(os.path.join('test', 'image_gt_d.tif'), img[0,:,:])
        # io.imsave(os.path.join('test', 'image_ori_d.tif'), img[3,:,:])
        # # # cv2.imwrite(os.path.join('test', 'image2.tif'), image[0,:,].detach().cpu().numpy().squeeze())
        # # # cv2.imwrite(os.path.join('test', 'gt.tiff'), gt[2].detach().cpu().numpy().squeeze())
        # image2_warped = image_warp(img[3,:,:], -np.transpose(flow[3,:,:,:], axes=(1,2,0))) # out H x W x C frame 2
        # io.imsave(os.path.join('test', 'image_warp_d.tif'), image2_warped)
        # cv2.imwrite('test/flow_d.tif', flow_to_image(np.transpose(flow[3,:,:,:], axes=(1,2,0))))

        # img1 = img[:, :, 0] # should be 512 x 512
        # img2 = img[:, :, 1] # should be 512 x 512

        # rotate the img and the flow
        # img = np.transpose(np.flip(img, axis=0), axes=(1,0,2))
        # flow = np.transpose(np.flip(flow, axis=0), axes=(1,0,2,3))

        # pre-processing
        img = preprocessing_img(img, self.norm_type)
        
        # cast to numpy
        # flow = np.array(flow).astype(np.float32)
        img = np.array(img).astype(np.float32)[:, np.newaxis,:,:]

        # Apply data augmentation
        if self.augmentor is not None:
            img = self.augmentor(img)

        # Convert to PyTorch tensors
        img = torch.from_numpy(img).float() # change to 8 x 1 x 512 x 512

        # debugging, copy and paste it to the debugger console
        # plot_images_and_flow(img1, img2, flow)

        # return img1, img2, flow, valid.float()
        # GT_image
        return img
        # mask_image
        # return img1, img2, flow, gt, mask, valid.float()

    def __del__(self):
        if hasattr(self, 'img_hdf5'):
            self.img_hdf5.close()

    def __rmul__(self, v):
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return self.length
