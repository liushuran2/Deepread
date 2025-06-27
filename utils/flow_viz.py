# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np


import matplotlib.pyplot as plt
import torch


def show_warped_changes(img1, img2):
    # img1: 1 x H x W, np array
    # img2: 1 x H x W, np array
    # purpose: 
    # 1. normalized both image based on the maximum value of both image, i.e. np.max(np.cat([img1, img2])), and normalized images are both in 0 and 1
    # 2. let img1 to be colorful array, represent color of cyan. let img2 to be colorful array, represent color of red
    # 3. let img1 to be the background, and img2 to be the foreground
    # 4. plot
    # test function
    # H, W = 100, 100
    # img1 = np.random.rand(1, H, W)  # Random noise image
    # img2 = np.random.rand(1, H, W)  # Another random noise image

    # # Call the function
    # show_warped_changes(img1, img2)

    # pre-step
    if len(img1.shape) != 3:
        img1 = img1[np.newaxis, ...]
    if len(img2.shape) != 3:
        img2 = img2[np.newaxis, ...]

    # Step 1: Normalize both images based on the maximum value of both images
    combined_min = np.min(np.concatenate([img1, img2]))
    img1 = img1 - combined_min
    img2 = img2 - combined_min
    combined_max = np.max(np.concatenate([img1, img2]))
    
    norm_img1 = img1 / combined_max
    norm_img2 = img2 / combined_max

    # Step 2: Apply color transformations
    # Create RGB arrays for each image
    # Cyan (0, 1, 1) and Red (1, 0, 0)
    img1_rgb = np.zeros((img1.shape[1], img1.shape[2], 3))  # Convert to RGB
    img2_rgb = np.zeros((img2.shape[1], img2.shape[2], 3))  # Convert to RGB

    # Set img1 as Cyan
    img1_rgb[:, :, 1] = norm_img1[0]  # Set Green channel
    img1_rgb[:, :, 2] = norm_img1[0]  # Set Blue channel

    # Set img2 as Red
    img2_rgb[:, :, 0] = norm_img2[0]  # Set Red channel

    # Step 3: Combine images with img1 as the background and img2 as the foreground
    # To overlap, simply add the two RGB images
    combined_img = img1_rgb + img2_rgb

    # # Step 4: Plot the result
    # plt.imshow(combined_img)
    # plt.axis('off')  # Hide axis for better visualization
    # plt.title("Warped Changes")
    # plt.show()
    return combined_img

def direction_plot_flow(flow1, flow2):
    # flow1: 2 x H x W, np array
    # flow2: 2 x H x W, np array

    combined_diff_x = show_warped_changes(flow1[0], flow2[0])
    combined_diff_y = show_warped_changes(flow1[1], flow2[1])
    return combined_diff_x, combined_diff_y


# current version of the tensorboard and wandb do not support table plotting
# def wandb_plot_cross_section():
#     # flow1: 2 x H x W, np array
#     # flow2: 2 x H x W, np array
#     # purpose:
#     # 1. get cross section of the flow1 and flow2, in the middle of the image, for both x and y (x and y represent first dimension of flow). 
#     #   thus we have totally 4 cross sections pairs (flow1_x_h, flow2_x_h), (flow1_x_w, flow2_x_w), (flow1_y_w, flow2_y_w), (flow1_y_h, flow2_y_h), where h and w are height and width of the image,
#     # 2. make the paired cross section, i.e. (flow1_x_h, flow2_x_h), to tables, mimicking `data = [[x, y] for (x, y) in zip(recall_micro, precision_micro)]`
#     # 3. plot the tables in wandb, i.e. `table = wandb.Table(data=data, columns = ["recall_micro", "precision_micro"])`, ``
#     pass

# visualize the flow, single image, for debugging purposes
def visualize_flow(images):
    # Assuming images is a numpy array of shape (512, 512, 3)
    height, width, channels = images.shape

    # Create a subplot for each image in the batch
    fig, axs = plt.subplots(1, 1, figsize=(20, 20))


    # Normalize to [0, 1] range if necessary
    img = images
    img = (img - img.min()) / (img.max() - img.min())

    # Display the image
    axs.imshow(img)
    axs.axis('off')

    plt.show()

# visualize the flow, batch version, for debugging purposes
def visualize_batch_flow(images):
    # Assuming images is a numpy array of shape (6, 512, 512, 3)
    batch_size, height, width, channels = images.shape

    # Create a subplot for each image in the batch
    fig, axs = plt.subplots(1, batch_size, figsize=(20, 20))

    for i in range(batch_size):
        # Normalize to [0, 1] range if necessary
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())

        # Display the image
        axs[i].imshow(img)
        axs[i].axis('off')

    plt.show()

# visualize the image and flow, for debugging purposes
def plot_images_and_flow(img1, img2, flow):
    img1_squeezed = img1.squeeze().numpy()
    img2_squeezed = img2.squeeze().numpy()

    # Normalize to [0, 1] range
    img1_squeezed = (img1_squeezed - img1_squeezed.min()) / (img1_squeezed.max() - img1_squeezed.min())
    img2_squeezed = (img2_squeezed - img2_squeezed.min()) / (img2_squeezed.max() - img2_squeezed.min())

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img1_squeezed, cmap='gray')
    axs[0].set_title('img1')

    axs[1].imshow(img2_squeezed, cmap='gray')
    axs[1].set_title('img2')

    # Display the magnitude of the flow for the third image
    flow_magnitude = torch.norm(flow, dim=0).numpy()
    axs[2].imshow(flow_magnitude, cmap='hot')
    axs[2].set_title('flow magnitude')

    plt.show()

# generate a color wheel for optical flow visualization, original RAFT code
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

# flow to colors, not normalized
def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v)) # they do a scaling here.
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

# Warning: this function will normalize the input flow
def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad) # do normalization
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)