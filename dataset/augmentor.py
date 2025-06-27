import numpy as np
import random
import math
from PIL import Image
from scipy.ndimage import zoom

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn.functional as F

#  modify for non-colorful images
class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        # erase augmentation params
        self.eraser_aug_prob = 0.3

    def eraser_transform(self, img1, img2, bounds=[25, 50]):
        """ 
        Occlusion augmentation 
        require img1 size: [h, w, c]
        """
        ht, wd = img1.shape[:2]
        c = img1.shape[2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, c), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color #

        return img1, img2

    def spatial_transform(self, img, flow):
        """ 
        Sptial transformation 
        require img1 size: [h, w, t, c]
        note in many case c can be 1
        """
        # randomly sample scale, x and y can be different
        t, c, ht, wd = img.shape
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)


        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img = zoom(img, (1, 1, scale_x, scale_y))
            
            flow = zoom(flow, (1, 1, scale_x, scale_y))

            factor = np.array([scale_x, scale_y])
            flow = flow * factor[None, :, None, None]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img = np.flip(img, axis=2)
                flow = np.flip(flow, axis=2)

                factor = np.array([1.0, -1.0])

                flow = flow * factor[None, :, None, None]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img = np.flip(img, axis=3)
                flow = np.flip(flow, axis=3)

                factor = np.array([-1.0, 1.0])

                flow = flow * factor[None, :, None, None]
        if img.shape[0] - self.crop_size[0] == 0:
            y0 = 0
            x0 = 0
        else:
            y0 = np.random.randint(0, img.shape[2] - self.crop_size[0])
            x0 = np.random.randint(0, img.shape[3] - self.crop_size[1])
        
        
        img = img[:,:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        flow = flow[:,:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        # make sure the size is correct
        if img.ndim== 3:
            img = img[..., np.newaxis]
            # mask = mask[..., np.newaxis]
        return img, flow
        # GT-image mask-image
        # return img1, img2, gt, flow, mask

    def __call__(self, img, flow):
        # img1, img2 = self.eraser_transform(img1, img2)
        # img1, img2, flow = self.spatial_transform(img1, img2, flow)

        #GT-image
        img, flow = self.spatial_transform(img, flow)

        img = np.ascontiguousarray(img)
        # img2 = np.ascontiguousarray(img2)

        # mask_image
        # mask = np.ascontiguousarray(mask)

        flow = np.ascontiguousarray(flow)

        # return img1, img2, flow
        return img, flow
        # return img1, img2, gt, flow, mask

#  modify for non-colorful images
class FinetuneAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        # erase augmentation params
        self.eraser_aug_prob = 0.3

    def spatial_transform(self, img):
        """ 
        Sptial transformation 
        require img1 size: [h, w, t, c]
        note in many case c can be 1
        """
        # randomly sample scale, x and y can be different
        t, c, ht, wd = img.shape
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)


        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img = zoom(img, (1, 1, scale_x, scale_y))
            
            flow = zoom(flow, (1, 1, scale_x, scale_y))

            factor = np.array([scale_x, scale_y])
            flow = flow * factor[None, :, None, None]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img = np.flip(img, axis=2)

                factor = np.array([1.0, -1.0])

            if np.random.rand() < self.v_flip_prob: # v-flip
                img = np.flip(img, axis=3)

                factor = np.array([-1.0, 1.0])

        if img.shape[0] - self.crop_size[0] == 0:
            y0 = 0
            x0 = 0
        else:
            y0 = np.random.randint(0, img.shape[2] - self.crop_size[0])
            x0 = np.random.randint(0, img.shape[3] - self.crop_size[1])
        
        
        img = img[:,:,y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        # make sure the size is correct
        if img.ndim== 3:
            img = img[..., np.newaxis]
            # mask = mask[..., np.newaxis]
        return img
        # GT-image mask-image
        # return img1, img2, gt, flow, mask

    def __call__(self, img):
        # img1, img2 = self.eraser_transform(img1, img2)
        # img1, img2, flow = self.spatial_transform(img1, img2, flow)

        #GT-image
        img = self.spatial_transform(img)

        img = np.ascontiguousarray(img)
        # img2 = np.ascontiguousarray(img2)

        # mask_image
        # mask = np.ascontiguousarray(mask)

        # return img1, img2, flow
        return img
        # return img1, img2, gt, flow, mask