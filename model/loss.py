
import torch

import numpy as np
from skimage.metrics import structural_similarity as ssim


# exclude extremly large displacements
def sequence_loss(flow_preds, flow_gt, data, valid, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions """
    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()

    n_predictions = len(flow_preds)    # batch size
    flow_loss = 0.0
    data_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow) # this is interesting

    for i in range(n_predictions): # for each prediction
        i_weight = gamma**(n_predictions - i - 1) # decay ratio
        flow_i_loss = (flow_preds[i] - flow_gt).abs() # absolute error
        data_i_loss = L1_pixelwise(data[i][0], data[i][1]) * 0.5 + L2_pixelwise(data[i][0], data[i][1]) * 0.5

        flow_loss += i_weight * (valid[:, None] * flow_i_loss).mean()
        data_loss += i_weight * (valid[:, None] * data_i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    # metrics, note it is a dictionary
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, data_loss, metrics

def sequence_data_loss(data, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions """
    L1_pixelwise = torch.nn.L1Loss()
    L2_pixelwise = torch.nn.MSELoss()

    n_predictions = len(data)    # batch size
    data_loss = 0.0

    for i in range(n_predictions): # for each prediction
        i_weight = gamma**(n_predictions - i - 1) # decay ratio
        data_i_loss = L1_pixelwise(data[i][0], data[i][1]) * 0.5 + L2_pixelwise(data[i][0], data[i][1]) * 0.5

        data_loss += i_weight * data_i_loss.mean()

    return data_loss

# Laplace_loss
def sequence_laplace_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    # batch size
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow) # this is interesting

    for i in range(n_predictions): # for each prediction
        i_weight = gamma**(n_predictions - i - 1) # decay ratio
        i_loss = (flow_preds[i] - flow_gt).abs() # absolute error
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    # metrics, note it is a dictionary
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

# exclude extremly large displacements
def sequence_maskedloss(flow_preds, flow_gt, valid, mask, gamma=0.8, max_flow=400):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    # batch size
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow) # this is interesting

    for i in range(n_predictions): # for each prediction
        i_weight = gamma**(n_predictions - i - 1) # decay ratio
        i_loss = ((flow_preds[i] - flow_gt) * mask).abs() # absolute error
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum(((flow_preds[i] - flow_gt) * mask)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    # metrics, note it is a dictionary
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


# TODO metrics: NMI, correlation, and others



# MMD loss
def mean_max_intensity_difference(raw_images, warped_images): # why this can work? curious
    """
    Calculate the mean of the maximum-intensity difference between two sets of images.

    Parameters:
    raw_images (np.array): A 3D numpy array of images (stacks).
    warped_images (np.array): A 3D numpy array of images (stacks) to compare against.

    Returns:
    float: Mean of the maximum-intensity differences.
    
    # Example usage:
    # Assuming `raw_images` and `warped_images` are already loaded and preprocessed to the correct dimensions
    # mean_difference = mean_max_intensity_difference(raw_images, warped_images)
    # print("Mean Max-Intensity Difference:", mean_difference)
    """

    # Normalize raw_images
    raw_images = raw_images - raw_images.min()
    raw_images = raw_images / raw_images.max() * 255
    
    # Normalize warped_images
    warped_images = warped_images - warped_images.min()
    warped_images = warped_images / warped_images.max() * 255
    
    # Calculate max intensity projections
    raw_max = np.max(raw_images, axis=0).astype(np.float32)
    warped_max = np.max(warped_images, axis=0).astype(np.float32)
    
    # Calculate the differences
    differences = np.abs(raw_max - warped_max)
    
    # Return the mean difference
    return np.mean(differences)


def mean_intensity_difference(template1, template2):
    """
    Calculate the mean intensity difference between two image templates.
    
    Parameters:
    template1, template2 (np.array): Two image templates for comparison.
    
    Returns:
    float: Mean intensity difference.
    """
    return np.mean(template1 - template2)

def calculate_rmse(template1, template2):
    """
    Calculate the root mean square error between two image templates.
    
    Parameters:
    template1, template2 (np.array): Two image templates for comparison.
    
    Returns:
    float: Root mean square error.
    """
    return np.sqrt(np.mean((template1 - template2) ** 2))

def calculate_mse(template1, template2):
    """
    Calculate the root mean square error between two image templates.
    
    Parameters:
    template1, template2 (np.array): Two image templates for comparison.
    
    Returns:
    float: Root mean square error.
    """
    return torch.mean((template1 - template2) ** 2)

def calculate_gradient_loss(template1, template2):
    """
    Calculate the root mean square error between two image templates.
    
    Parameters:
    template1, template2 (np.array): Two image templates for comparison.
    
    Returns:
    float: Root mean square error.
    """
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()  # (1, 1, 3, 3)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()  # (1, 1, 3, 3)
    
    # compute gradient
    gradient_x_1 = torch.nn.functional.conv2d(template1, sobel_x, padding=1)  # (batch, 1, h, w)
    gradient_y_1 = torch.nn.functional.conv2d(template1, sobel_y, padding=1)  # (batch, 1, h, w)
    gradient_x_2 = torch.nn.functional.conv2d(template2, sobel_x, padding=1)  # (batch, 1, h, w)
    gradient_y_2 = torch.nn.functional.conv2d(template2, sobel_y, padding=1)  # (batch, 1, h, w)

    # concat
    gradient1 = torch.cat((gradient_x_1, gradient_y_1), dim=1)  # (batch, 2, h, w)
    gradient2 = torch.cat((gradient_x_2, gradient_y_2), dim=1)  # (batch, 2, h, w)

    return torch.mean((gradient1 - gradient2) ** 2)

def calculate_ssim(template1, template2):
    """
    Calculate the structural similarity index between two image templates.
    
    Parameters:
    template1, template2 (np.array): Two image templates for comparison.
    
    Returns:
    float: Structural similarity index.
    """
    return ssim(template1, template2, data_range=template1.max() - template1.min())

# Usage of metrics for motion registration assessment
def assess_motion_registration(raw_images, registered_images, new_images, normcorre_images, patchwarp_images):
    templates = {
        "raw": np.mean(raw_images, axis=0),
        "registered": np.mean(registered_images, axis=0),
        "new": np.mean(new_images, axis=0),
        "normcorre": np.mean(normcorre_images, axis=0),
        "patchwarp": np.mean(patchwarp_images, axis=0)
    }
    
    metrics = {}
    for key1 in templates:
        for key2 in templates:
            if key1 != key2:
                mi_diff = mean_intensity_difference(templates[key1], templates[key2])
                rmse_val = calculate_rmse(templates[key1], templates[key2])
                ssim_val = calculate_ssim(templates[key1], templates[key2])
                metrics[(key1, key2)] = (mi_diff, rmse_val, ssim_val)
    
    return metrics

# Example of how you might initialize image data and call the function
# raw_images, registered_images, new_images, normcorre_images, patchwarp_images = load_your_images()
# results = assess_motion_registration(raw_images, registered_images, new_images, normcorre_images, patchwarp_images)
# print(results)



# correlation loss, added in 2024.05.01

# def pearson_correlation(imageA, imageB):
#     #
#     """
#     Calculate the Pearson correlation coefficient between two images.
    
#     Args:
#     - imageA (torch.Tensor): Tensor of shape (1, 1, T, H, W)
#     - imageB (torch.Tensor): Tensor of shape (1, 1, T, H, W)
    
#     Returns:
#     - torch.Tensor: Pearson correlation coefficient for each pixel across all frames

#     # Example usage:
#     # Create dummy data for imageA and imageB
#     imageA = torch.randn(1, 1, 10, 256, 256)  # Random tensor simulating an image sequence
#     imageB = torch.randn(1, 1, 10, 256, 256)  # Another random tensor

#     # Calculate Pearson correlation
#     correlation_map = pearson_correlation(imageA, imageB)
#     print(correlation_map.shape)  # Should be (256, 256), showing correlation at each pixel

#     """
#     # Ensure the input tensors are float type
#     imageA = imageA.float()
#     imageB = imageB.float()
    
#     # Reshape the images to collapse all dimensions except the last two (H, W)
#     # This makes each pixel in HxW an independent variable across the T dimension
#     imageA = imageA.view(-1, imageA.shape[-2], imageA.shape[-1])
#     imageB = imageB.view(-1, imageB.shape[-2], imageB.shape[-1])
    
#     # Compute means along the first dimension (T)
#     meanA = imageA.mean(dim=0, keepdim=True)
#     meanB = imageB.mean(dim=0, keepdim=True)
    
#     # Compute deviations from means
#     devA = imageA - meanA
#     devB = imageB - meanB
    
#     # Compute covariance between deviations
#     cov = (devA * devB).mean(dim=0)
    
#     # Compute standard deviations of both images
#     stdA = devA.pow(2).mean(dim=0).sqrt()
#     stdB = devB.pow(2).mean(dim=0).sqrt()
    
#     # Compute Pearson correlation coefficient
#     corr = cov / (stdA * stdB)
    
#     return corr

def pearson_corr(tensor1, tensor2):
    # Reshape tensors to (Batch, Height * Width)
    tensor1_flat = tensor1.view(tensor1.shape[0], -1)  # (Batch, Height * Width)
    tensor2_flat = tensor2.view(tensor2.shape[0], -1)  # (Batch, Height * Width)

    # 计算每个样本的均值
    mean1 = tensor1_flat.mean(dim=1, keepdim=True)
    mean2 = tensor2_flat.mean(dim=1, keepdim=True)

    # 减去均值
    tensor1_centered = tensor1_flat - mean1
    tensor2_centered = tensor2_flat - mean2

    # 计算分子和分母
    numerator = (tensor1_centered * tensor2_centered).sum(dim=1)
    denominator = torch.sqrt((tensor1_centered ** 2).sum(dim=1) * (tensor2_centered ** 2).sum(dim=1))

    # 防止分母为零
    correlation = numerator / (denominator + 1e-8)

    return correlation
