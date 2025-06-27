import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import random
import json
import argparse
import yaml

# load json
class FlexibleNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def load_from_json(cls, filename):
        with open(filename, 'r') as f:
            args_dict = json.load(f)
        # Create an instance of cls (which is FlexibleNamespace here) with the loaded arguments
        return cls(**args_dict)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

# load yaml
class FlexibleNamespace_yml(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def load_from_yml(cls, filename):
        with open(filename, 'r') as f:
            args_dict = yaml.safe_load(f)
        # Create an instance of cls (which is FlexibleNamespace here) with the loaded arguments
        return cls(**args_dict)

    def __setattr__(self, name, value):
        self.__dict__[name] = value


def save_args(args, filename='args.json'):
    args_dict = vars(args)

    # Convert NumPy types to standard Python types
    for key, value in args_dict.items():
        if isinstance(value, np.float32):
            args_dict[key] = float(value)  
        elif isinstance(value, np.ndarray):  # Handle other NumPy arrays
            args_dict[key] = value.tolist() 

    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)

# helper function
def print_args_formatted(args, name_prefix=""):
    """Prints arguments and nested structures in a consistent format, sorted alphabetically.

    Args:
        args: The namespace object or dictionary containing arguments.
        name_prefix: An optional prefix to prepend to variable names.
    """

    sorted_attrs = sorted(args.__dict__.items())  # Sort attributes alphabetically

    for attr, value in sorted_attrs:
        full_name = f"{name_prefix}{attr}"

        if isinstance(value, (dict, argparse.Namespace)):
            # Handle nested structures recursively
            print_args_formatted(value, f"{full_name}.") # recursive call
        else:
            # Print simple key-value pairs
            print(f"{full_name} = {value}")


# this is for dataloader and reproducibility
def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


# my favourite
def data_to_gpu(x, device):
    return x.to(device)


def RoundTensor(x, num=2, group_str=False):
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            str_list.append(','.join(x_row))
        out_str = '/'.join(str_list)
    else:
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str


###
def read_key_file(filepath):
    with open(filepath, 'r') as f:
        key = f.read().strip()  # Remove any extra whitespace
    return key

###
# nan detection
def has_nan_weights(model):
    """Checks if a PyTorch model has any NaN weights or biases."""
    for name, param in model.named_parameters():
        if torch.any(torch.isnan(param)):
            print(f"Parameter '{name}' contains NaN values.")
            return True
    return False  # No NaNs found

def has_nan_in_dict(weight_dict):
    """Checks if a dictionary of weights contains any NaN values."""
    for key, tensor in weight_dict.items():
        if torch.any(torch.isnan(tensor)):
            print(f"Tensor '{key}' contains NaN values.")
            return True
    return False  # No NaNs found

#########################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



    
# logger from original RAFT code
# class Logger:
#     def __init__(self, model, scheduler):
#         self.model = model
#         self.scheduler = scheduler
#         self.total_steps = 0
#         self.running_loss = {}
#         self.writer = None

#     def _print_training_status(self):
#         metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
#         training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
#         metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
#         # print the training status
#         print(training_str + metrics_str)

#         if self.writer is None:
#             self.writer = SummaryWriter()

#         for k in self.running_loss:
#             self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
#             self.running_loss[k] = 0.0

#     def push(self, metrics):
#         self.total_steps += 1

#         for key in metrics:
#             if key not in self.running_loss:
#                 self.running_loss[key] = 0.0

#             self.running_loss[key] += metrics[key]

#         if self.total_steps % SUM_FREQ == SUM_FREQ-1:
#             self._print_training_status()
#             self.running_loss = {}

#     def write_dict(self, results):
#         if self.writer is None:
#             self.writer = SummaryWriter()

#         for key in results:
#             self.writer.add_scalar(key, results[key], self.total_steps)

#     def close(self):
#         self.writer.close()



# original functions. modified
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8

        self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

# 
def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()

# sample the iamge based on the grid
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True) # sample the image using the grid

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

# build coords grids
def coords_grid(batch, ht, wd, device):
    # height (ht) and width (wd) are the dimensions of the image
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
    # output is a tensor of shape [batch, 2, ht, wd]

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def gaussian_blur(input_tensor, kernel_size=3, sigma=2.0):
    # 创建高斯核
    kernel = torch.tensor([
        [1/(2 * 3.14159 * sigma**2) * torch.exp(torch.tensor(-((x - kernel_size // 2)**2 + (y - kernel_size // 2)**2) / (2 * sigma**2)))
         for x in range(int(kernel_size))] for y in range(kernel_size)]).cuda()
    
    kernel = kernel / kernel.sum()  # 归一化
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 改变形状为 (1, 1, kernel_size, kernel_size)

    # 使用F.conv2d进行卷积
    smoothed_tensor = F.conv2d(input_tensor, kernel, padding=kernel_size // 2)

    return smoothed_tensor

def bin_median(mat, window=10, exclude_nans=True):
    """ compute median of 3D array in along axis o by binning values

    Args:
        mat: ndarray
            input 3D matrix, time along first dimension

        window: int
            number of frames in a bin

    Returns:
        img:
            median image

    Raises:
        Exception 'Path to template does not exist:'+template
    """

    T, d1, d2 = np.shape(mat)
    if T < window:
        window = T
    num_windows = int(T // window)
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    return img
