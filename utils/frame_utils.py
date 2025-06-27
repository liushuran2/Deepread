import numpy as np
from PIL import Image
from os.path import *
import re
import math
import torch
from skimage import io
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from types import FunctionType

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

# read optical flow from file, .flo file
def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()



# read the image
def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    return []

# do image warp
# adopted from yuhong
def image_warp(im, flow, mode='bilinear'):
    # code adopted from yuhong
    """Performs a backward warp of an image using the predicted flow.
    numpy version

    Args:
        im: input image. ndim=2, 3 or 4, [[num_batch], height, width, [channels]]. num_batch and channels are optional, default is 1.
        flow: flow vectors. ndim=3 or 4, [[num_batch], height, width, 2]. num_batch is optional
        mode: interpolation mode. 'nearest' or 'bilinear'
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    # assert im.ndim == flow.ndim, 'The dimension of im and flow must be equal '
    flag = 4
    if im.ndim == 2:
        height, width = im.shape
        num_batch = 1
        channels = 1
        im = im[np.newaxis, :, :, np.newaxis]
        flow = flow[np.newaxis, :, :]
        flag = 2
    elif im.ndim == 3:
        height, width, channels = im.shape
        num_batch = 1
        im = im[np.newaxis, :, :]
        flow = flow[np.newaxis, :, :]
        flag = 3
    elif im.ndim == 4:
        num_batch, height, width, channels = im.shape
        flag = 4
    else:
        raise AttributeError('The dimension of im must be 2, 3 or 4')

    max_x = width - 1
    max_y = height - 1
    zero = 0

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = np.reshape(im, [-1, channels])
    flow_flat = np.reshape(flow, [-1, 2])


    # Floor the flow, as the final indices are integers
    flow_floor = np.floor(flow_flat).astype(np.int32) # floor the flow to int

    # Construct base indices which are displaced with the flow
    pos_x = np.tile(np.arange(width), [height * num_batch])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])
    pos_y = np.tile(np.reshape(grid_y, [-1]), [num_batch])

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]

    x0 = pos_x + x
    y0 = pos_y + y

    x0 = np.clip(x0, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    # x_idx = np.where((x0 > max_x) | (x0 < 0))
    # y_idx = np.where((y0 > max_y) | (y0 < 0))
    # x0[x_idx] = pos_x[x_idx]
    # y0[y_idx] = pos_y[y_idx]

    dim1 = width * height
    batch_offsets = np.arange(num_batch) * dim1
    base_grid = np.tile(np.expand_dims(batch_offsets, 1), [1, dim1])
    base = np.reshape(base_grid, [-1])

    base_y0 = base + y0 * width

    if mode == 'nearest':
        idx_a = base_y0 + x0
        warped_flat = im_flat[idx_a]
    elif mode == 'bilinear':
        # The fractional part is used to control the bilinear interpolation.
        bilinear_weights = flow_flat - np.floor(flow_flat)

        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = np.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = np.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = np.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = np.expand_dims(xw * yw, 1) # bottom right pixel

        x1 = x0 + 1
        y1 = y0 + 1

        x1 = np.clip(x1, zero, max_x)
        y1 = np.clip(y1, zero, max_y)

        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        warped_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    warped = np.reshape(warped_flat, [num_batch, height, width, channels])

    if flag == 2:
        warped = np.squeeze(warped)
    elif flag == 3:
        warped = np.squeeze(warped, axis=0)
    else:
        pass

    warped = warped.astype(np.float32)
    return warped


#############################
# preprocessing video
import numpy as np

def preprocessing_img(video, normalization_methods):
    # Handle zero-size array
    
    if video.size == 0:
        return np.array([])  # Or appropriate error/message
    
    # Initial check for NaN and Inf values
    if np.isnan(video).any() or np.isinf(video).any():
        # Optional: handle or replace NaN and Inf values here
        video = np.nan_to_num(video, nan=np.nanmean(video), posinf=np.max(np.nan_to_num(video, nan=np.nanmean(video))), 
                              neginf=np.min(np.nan_to_num(video, nan=np.nanmean(video))))
    
    if normalization_methods == 'max':
        normalized_video = video / np.max(video)
        
    elif normalization_methods == 'min_max':
        range = np.max(video) - np.min(video)
        if range == 0:  # Prevent division by zero
            return np.zeros(video.shape)
        normalized_video = (video - np.min(video)) / range
        
    elif normalization_methods == 'mean':
        normalized_video = video - np.mean(video)
        
    elif normalization_methods == 'std':
        std = np.std(video)
        if std == 0:  # Prevent division by zero
            return np.zeros(video.shape)
        normalized_video = (video - np.mean(video)) / std
        
    elif normalization_methods == 'mean_max':
        mean = np.mean(video)
        max_minus_mean = np.max(video) - mean
        
        if max_minus_mean == 0:  # Prevent division by zero
            return np.zeros(video.shape)
        normalized_video = (video - mean) / max_minus_mean
        
    elif normalization_methods == 'robust':
        p1 = np.percentile(video, 1)
        p99 = np.percentile(video, 99)
        range = p99 - p1
        if range == 0:  # Prevent division by zero
            return np.zeros(video.shape)
        normalized_video = (video - p1) / range
        # normalized_video = np.clip(normalized_video, 0, 1) # this should not be included
    else:
        raise ValueError("Unsupported normalization method.")
    

    return normalized_video


# post processing the video
def postprocessing_video(normalized_video, normalization_methods, data_property):
    # normalized_video: normalized video
    # normalization_methods: normalization method
    # data_property: dictionary containing all necessary statistics
    
    # Handle zero-size array
    if normalized_video.size == 0:
        return np.array([])  # Or appropriate error/message
    
    # Assuming data_property dictionary contains all necessary statistics
    if normalization_methods == 'max':
        video = normalized_video * data_property['max']
    elif normalization_methods == 'min_max':
        range = data_property['max'] - data_property['min']
        video = (normalized_video * range) + data_property['min']
    elif normalization_methods == 'mean':
        video = normalized_video + data_property['mean']
    elif normalization_methods == 'std':
        video = (normalized_video * data_property['std']) + data_property['mean']
    elif normalization_methods == 'mean_max':
        max_minus_mean = data_property['max'] - data_property['mean']
        video = (normalized_video * max_minus_mean) + data_property['mean']
    elif normalization_methods == 'robust':
        range = data_property['p99'] - data_property['p1']
        video = (normalized_video * range) + data_property['p1']
    else:
        raise ValueError("Unsupported normalization method.")
    
    # No need to clip in post-processing unless specifically desired for 'robust'
    # If clipping was applied during preprocessing for a specific reason, reconsider if it should be applied here.
    
    
    return clip_vdieo(video)


#############################
# training eception handling
def clip_vdieo(video):
    # clip the nan, posinf, and neginf
    # if all nan, replace with 0

    clip_video = np.nan_to_num(video, nan=np.nanmean(video), 
                                 posinf=np.max(np.nan_to_num(video, nan=np.nanmean(video))), 
                                 neginf=np.min(np.nan_to_num(video, nan=np.nanmean(video))))   
   # Check if all elements are NaN in any frame
    for frame in clip_video:
        if np.all(np.isnan(frame)):  # If all elements in a frame are NaN
            frame[:] = 0  # Replace the entire frame with 0
    return clip_video

def clip_video_torch(video):
    # Assuming video is a PyTorch tensor
    if not torch.is_tensor(video):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Calculate mean, max, and min with NaN values ignored
    # nanmean = torch.nanmean(video) # this is not avaliable in 3090
    nanmean = torch.mean(video[~video.isnan()])
    nanmax = torch.max(torch.nan_to_num(video, nan=float('-inf')))
    nanmin = torch.min(torch.nan_to_num(video, nan=float('inf')))
    
    # Replace NaN with nanmean, posinf with nanmax, and neginf with nanmin
    video = torch.where(torch.isnan(video), nanmean, video)
    video = torch.clamp(video, min=nanmin, max=nanmax)
    # Handle all-NaN frames (optional, assuming frames are in the last dimension)
    if video.ndim > 1:
        for i in range(video.shape[-1]):
            frame = video[..., i]
            if torch.all(torch.isnan(frame)):
                frame[:] = 0
    return video


#############################
# save image
def save_image(output_img, input_data_type, result_name):
    
    if input_data_type == 'uint16':
        output_img=np.clip(output_img, 0, 65535)
        output_img = output_img.astype('uint16')

    elif input_data_type == 'int16':
        output_img=np.clip(output_img, -32767, 32767)
        output_img = output_img.astype('int16')

    elif input_data_type == 'uint8':
        output_img=np.clip(output_img, 0, 255)
        output_img = output_img.astype('uint8')

    else:
        output_img = output_img.astype('int32')
    
    io.imsave(result_name, output_img, check_contrast=False)
        

#############################
# save flow, as tensor

@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(make_grid)
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            t = t.float()  # convert to float
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid



def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")


def image_warp_tensor(im, flow, mode='bilinear'):
    # code adopted from yuhong
    """Performs a backward warp of an image using the predicted flow.
    numpy version

    Args:
        im: input image. ndim=2, 3 or 4, [[num_batch], height, width, [channels]]. num_batch and channels are optional, default is 1.
        flow: flow vectors. ndim=3 or 4, [[num_batch], height, width, 2]. num_batch is optional
        mode: interpolation mode. 'nearest' or 'bilinear'
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    # assert im.ndim == flow.ndim, 'The dimension of im and flow must be equal '
    if im.ndim == 2:
        height, width = im.shape
        num_batch = 1
        channels = 1
        im = im.unsqueeze(0).unsqueeze(-1)  # (1, height, width, 1)
        flow = flow.unsqueeze(0)  # (1, height, width, 2)
    elif im.ndim == 3:
        height, width, channels = im.shape
        num_batch = 1
        im = im[np.newaxis, :, :]
        flow = flow[np.newaxis, :, :]
    elif im.ndim == 4:
        num_batch, height, width, channels = im.shape
    else:
        raise AttributeError('The dimension of im must be 2, 3 or 4')

    max_x = width - 1
    max_y = height - 1
    zero = 0

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = im.reshape(-1, channels)  # (num_batch * height * width, channels)
    flow_flat = flow.reshape(-1, 2)     # (num_batch * height * width, 2)


    # Floor the flow, as the final indices are integers
    flow_floor = torch.floor(flow_flat).to(torch.int32)

    # Construct base indices which are displaced with the flow
    pos_x = torch.arange(width).repeat(height * num_batch).cuda()
    grid_y = torch.arange(height).view(-1, 1).repeat(1, width).view(-1)
    pos_y = grid_y.repeat(num_batch).cuda()

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]

    x0 = pos_x + x
    y0 = pos_y + y

    x0 = torch.clamp(pos_x + x, min=zero, max=max_x)
    y0 = torch.clamp(pos_y + y, min=zero, max=max_y)

    dim1 = width * height
    batch_offsets = torch.arange(num_batch) * dim1
    base_grid = batch_offsets.view(-1, 1).expand(-1, dim1).reshape(-1).cuda()

    base_y0 = base_grid + y0 * width

    if mode == 'nearest':
        idx_a = base_y0 + x0
        warped_flat = im_flat[idx_a]
    elif mode == 'bilinear':
        # The fractional part is used to control the bilinear interpolation.
        bilinear_weights = flow_flat - flow_floor.to(torch.float32)

        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = (1 - xw) * (1 - yw)  # top-left pixel
        wb = (1 - xw) * yw        # bottom-left pixel
        wc = xw * (1 - yw)        # top-right pixel
        wd = xw * yw              # bottom-right pixel

        x1 = torch.clamp(x0 + 1, min=zero, max=max_x)
        y1 = torch.clamp(y0 + 1, min=zero, max=max_y)

        base_y1 = base_grid + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        warped_flat = wa.view(-1, 1) * Ia + wb.view(-1, 1) * Ib + wc.view(-1, 1) * Ic + wd.view(-1, 1) * Id
    warped = warped_flat.view(num_batch, height, width, channels)

    # Remove added dimensions if needed
    if im.ndim == 2:
        warped = warped.squeeze()
    elif im.ndim == 3:
        warped = warped.squeeze(0)

    return warped.to(torch.float32)

def adjust_frame_intensity(video):
    # 获取视频的形状 (帧数, 高度, 宽度)
    num_frames, height, width = video.shape
    
    # 计算每一帧的平均强度
    frame_means = np.mean(video, axis=(1, 2))  # shape: (num_frames,)
    
    # 计算全局平均强度
    global_mean = np.mean(frame_means)
    
    # 调整每一帧的强度，使其平均值为 global_mean
    frame_means_factor = frame_means / global_mean
    adjusted_video = video / frame_means_factor[:, None, None]
    
    return adjusted_video