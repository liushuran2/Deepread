import scipy
from skimage import io
import numpy as np
from scipy.optimize import leastsq
from skimage.metrics import structural_similarity as ssim
import math
import csv
import os
import cv2


def linear_error(p,gt,pre):
    return p[0]*pre+p[1]-gt
    
def linear_trans(gt,pre):
    p0 = [1,0]
    result = leastsq(linear_error,p0,args=(gt.ravel(),pre.ravel()))
    a,b = result[0]
    return a*pre+b


def compute_metric(original_movie, reg_movie):
    # smoooth
    smoothness = np.sqrt(
            np.sum(np.sum(np.array(np.gradient(np.mean(reg_movie, 0)))**2, 0)))
    print(smoothness)

    # CM
    templ = np.median(original_movie, axis=0)
    templ = original_movie[0]

    correlations = []

    for fr in reg_movie:
        correlations.append(scipy.stats.pearsonr(
            fr.flatten(), templ.flatten())[0])

    print(np.mean(correlations))
        
    #mCD
    pre_max = np.max(original_movie, axis=0)
    post_max = np.max(reg_movie, axis=0)
    mCD = np.mean(post_max-pre_max)
    print(mCD)

    # PSNR,SSIM
    PSNRs = []
    SSIMs = []

    for fr in reg_movie:
        PSNRs.append(-10 * math.log10(np.mean((linear_trans(templ, fr) - templ) ** 2)))
        # SSIMs.append(ssim(linear_trans(templ, fr), templ))

    print(np.mean(PSNRs))
    # print(np.mean(SSIMs))

    # STD
    pre_std = np.mean(np.std(original_movie, axis=0))
    post_std = np.mean(np.std(reg_movie, axis=0))
    STD_ratio = pre_std / post_std
    print(STD_ratio)

    return [smoothness, np.mean(correlations), mCD, np.mean(PSNRs), STD_ratio]


original_movie = io.imread('/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test2/N_100_scale_10_noise.tiff')
reg_movie = io.imread('RAFT_result_frameloss05/N_100_scale_10_noise_reg.tiff')
original_movie= (original_movie - np.percentile(original_movie, 1)) / (np.percentile(original_movie, 99) - np.percentile(original_movie, 1))
reg_movie= (reg_movie - np.percentile(reg_movie, 1)) / (np.percentile(reg_movie, 99) - np.percentile(reg_movie, 1))

compute_metric(original_movie, reg_movie)
# all_result = []
# for id in range(10):
#     directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/2p_148d/trial_2p_' + str(id + 1) + '/motion/'
#     all_files = os.listdir(directory_path)

#     tiff_files = [filename for filename in all_files if filename.endswith('.tiff')]

#     count = 1

#     for fnames in tiff_files:
#         original_movie = io.imread(directory_path + fnames)
#         reg_movie = io.imread(directory_path + 'flow_reg/block_' + str(count).rjust(3, '0') + '/compensated.TIFF')
#         reg_movie = reg_movie[:, 21:491, 21:491]
#         original_movie = original_movie[:, 21:491, 21:491]
#         original_movie= (original_movie - np.min(original_movie)) / (np.max(original_movie) - np.min(original_movie))
#         reg_movie= (reg_movie - np.min(reg_movie)) / (np.max(reg_movie) - np.min(reg_movie))
#         reg_movie = linear_trans(original_movie, reg_movie)


#         result_list = compute_metric(original_movie, reg_movie)
#         all_result.append(result_list)

#         count = count + 1

# all_result = np.array(all_result)
# data = {'matrix': all_result}
# scipy.io.savemat('matrix.mat', data)

all_result = []
# mask_path = '/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/NeuronMask.mat'
for id in range(15):
    original_movie = io.imread('/mnt/nas/YZ_personal_storage/Private/MC/Gen_motion_simulation/test_seq/N_100_scale_10_noise_{}.tiff'.format(str(id + 1)))
    reg_movie = io.imread('RAFT_result/N_100_scale_10_noise_{}_reg.tiff'.format(str(id + 1)))
    # mask = scipy.io.loadmat(mask_path)['outputMask']

    original_movie= (original_movie - np.percentile(original_movie, 1)) / (np.percentile(original_movie, 99) - np.percentile(original_movie, 1))
    reg_movie= (reg_movie - np.percentile(reg_movie, 1)) / (np.percentile(reg_movie, 99) - np.percentile(reg_movie, 1))

    # original_movie_mask = original_movie * mask[np.newaxis, ...]
    # reg_movie_mask = reg_movie * mask[np.newaxis, ...]
    # cv2.imwrite(os.path.join('test', 'reg_movie_mask.tiff'), reg_movie_mask[5])

    # reg_movie = linear_trans(original_movie, reg_movie)

    result_list = compute_metric(original_movie, reg_movie)
    all_result.append(result_list)

all_result = np.array(all_result)
data = {'matrix': all_result}
scipy.io.savemat('matrix.mat', data)



