# %%
import sys
import os
import random
import time
import glob
import numpy as np
import math
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp

sys.path.insert(1, '../..') # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Set which GPU to use. '-1' uses only CPU.

from suns.PreProcessing.preprocessing_functions import preprocess_video, find_dataset
from suns.PreProcessing.generate_masks import generate_masks
from suns.train_CNN_params_varysize import train_CNN, parameter_optimization_cross_validation

import tensorflow as tf
tf_version = int(tf.__version__[0])
if tf_version == 1:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config = config)
else: # tf_version == 2:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# %%
if __name__ == '__main__':
    #-------------- Start user-defined parameters --------------#
    # %% set folders
    # file names of the ".h5" files storing the raw videos. 
    list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22'] 
    # folder of the raw videos
    dir_video = '../data' 
    # folder of the ".mat" files stroing the GT masks in sparse 2D matrices. 'FinalMasks_' is a prefix of the file names. 
    dir_GTMasks = os.path.join(dir_video, 'GT Masks', 'FinalMasks_') 
    
    # %% set video parameters
    list_rate_hz = [10] * len(list_Exp_ID) # frame rate of all the videos. Close frame rates are preferred.
    list_Mag = [6/8] * len(list_Exp_ID) # spatial magnification compared to ABO videos (0.785 um/pixel). # Mag = 0.785 / pixel_size

    # %% set the range of post-processing hyper-parameters to be optimized in
    # minimum area of a neuron (unit: pixels in ABO videos). must be in ascend order
    list_minArea = list(range(30,85,5)) 
    # average area of a typical neuron (unit: pixels in ABO videos)
    list_avgArea = [177] 
    # uint8 threshould of probablity map (uint8 variable, = float probablity * 256 - 1)
    list_thresh_pmap = list(range(130,235,10))
    # threshold to binarize the neuron masks. For each mask, 
    # values higher than "thresh_mask" times the maximum value of the mask are set to one.
    thresh_mask = 0.5
    # maximum COM distance of two masks to be considered the same neuron in the initial merging (unit: pixels in ABO videos)
    thresh_COM0 = 2
    # maximum COM distance of two masks to be considered the same neuron (unit: pixels in ABO videos)
    list_thresh_COM = list(np.arange(4, 9, 1)) 
    # minimum IoU of two masks to be considered the same neuron
    list_thresh_IOU = [0.5] 
    # minimum consecutive number of frames of active neurons
    list_cons = list(range(1, 8, 1)) 

    # dictionary of all fixed and searched post-processing parameters.
    Params_set = {'list_minArea': list_minArea, 'list_avgArea': list_avgArea, 'list_thresh_pmap': list_thresh_pmap,
            'thresh_COM0': thresh_COM0, 'list_thresh_COM': list_thresh_COM, 'list_thresh_IOU': list_thresh_IOU,
            'thresh_mask': thresh_mask, 'list_cons': list_cons}
    print(Params_set)

    # %% set training parameters
    thred_std = 3 # SNR threshold used to determine when neurons are active
    num_train_per = 2400 # Number of frames per video used for training 
    NO_OF_EPOCHS = 200 # Number of epoches used for training 
    batch_size_eval = 100 # batch size in CNN inference
    list_thred_ratio = [thred_std] # A list of SNR threshold used to determine when neurons are active.

    # %% Set processing options
    useSF=False # True if spatial filtering is used in pre-processing.
    useTF=True # True if temporal filtering is used in pre-processing.
    useSNR=True # True if pixel-by-pixel SNR normalization filtering is used in pre-processing.
    med_subtract=False # True if the spatial median of every frame is subtracted before temporal filtering.
        # Can only be used when spatial filtering is not used. 
    prealloc=False # True if pre-allocate memory space for large variables in pre-processing. 
            # Achieve faster speed at the cost of higher memory occupation.
            # Not needed in training.
    useWT=False # True if using additional watershed
    load_exist=False # True if using temp files already saved in the folders
    use_validation = True # True to use a validation set outside the training set
    useMP = True # True to use multiprocessing to speed up
    BATCH_SIZE = 20 # Batch size for training 
    # Cross-validation strategy. Can be "leave_one_out", "train_1_test_rest", or "use_all"
    cross_validation = "leave_one_out"
    Params_loss = {'DL':1, 'BCE':20, 'FL':0, 'gamma':1, 'alpha':0.25} # Parameters of the loss function
    #-------------- End user-defined parameters --------------#


    dir_parent = os.path.join(dir_video, 'noSF_multi_size') # folder to save all the processed data
    dir_network_input = os.path.join(dir_parent, 'network_input') # folder of the SNR videos
    dir_mask = os.path.join(dir_parent, 'temporal_masks({})'.format(thred_std)) # foldr to save the temporal masks
    weights_path = os.path.join(dir_parent, 'Weights') # folder to save the trained CNN
    training_output_path = os.path.join(dir_parent, 'training output') # folder to save the loss functions during training
    dir_output = os.path.join(dir_parent, 'output_masks') # folder to save the optimized hyper-parameters
    dir_temp = os.path.join(dir_parent, 'temp') # temporary folder to save the F1 with various hyper-parameters

    if not os.path.exists(dir_network_input):
        os.makedirs(dir_network_input) 
    if not os.path.exists(weights_path):
        os.makedirs(weights_path) 
    if not os.path.exists(training_output_path):
        os.makedirs(training_output_path) 
    if not os.path.exists(dir_output):
        os.makedirs(dir_output) 
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp) 

    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    # Get and check the dimensions of all the videos
    list_Dimens3 = np.zeros((nvideo, 3),dtype='uint16')
    for (eid,Exp_ID) in enumerate(list_Exp_ID):
        h5_video = os.path.join(dir_video, Exp_ID + '.h5')
        h5_file = h5py.File(h5_video,'r')
        dset = find_dataset(h5_file)
        list_Dimens[eid] = h5_file[dset].shape
        h5_file.close()

    list_nframes = list_Dimens3[:,0]
    list_Dimens = [x[1:] for x in list_Dimens3]
    list_TF_size = np.zeros(nvideo, dtype='uint16')


    # pre-processing for training
    for (ind_video, Exp_ID) in enumerate(list_Exp_ID): # 
        rate_hz = list_rate_hz[ind_video]
        Mag = list_Mag[ind_video]
        num_median_approx = 1000 # number of frames used to caluclate median and median-based standard deviation

        # %% set temporal filter
        if useTF:
            filename_TF_template = '../YST_spike_tempolate.h5' # File name storing the temporal filter kernel
            h5f = h5py.File(filename_TF_template,'r')
            Poisson_filt = np.array(h5f['filter_tempolate']).squeeze().astype('float32')
            h5f.close()

            # Rescale the filter template according to "rate_hz"
            # It assumes the calcium sensors are the same, but the frame rates are different
            fs_template = 10 # frame rate of the filter tempolate
            peak = Poisson_filt.argmax()
            length = Poisson_filt.shape
            xp = np.arange(-peak,length-peak,1)/fs_template
            x = np.arange(np.round(-peak*rate_hz/fs_template), np.round(length-peak*rate_hz/fs_template), 1)/rate_hz
            Poisson_filt = np.interp(x,xp,Poisson_filt).astype('float32')
            
            Poisson_filt = Poisson_filt[Poisson_filt>np.exp(-1)] # temporal filter kernel
            Poisson_filt = Poisson_filt/Poisson_filt.sum()

            # # Alternative temporal filter kernel using a single exponential decay function
            # decay = 0.8 # decay time constant (unit: second)
            # leng_tf = np.ceil(rate_hz*decay)+1
            # Poisson_filt = np.exp(-np.arange(leng_tf)/rate_hz/decay)
            # Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype('float32')
        else:
            Poisson_filt=np.array([1], dtype='float32')
        list_TF_size[ind_video] = Poisson_filt.size

        # dictionary of pre-processing parameters
        gauss_filt_size = 50*Mag # standard deviation of the spatial Gaussian filter in pixels
        Params_pre = {'gauss_filt_size':gauss_filt_size, 'num_median_approx':num_median_approx, 
            'Poisson_filt': Poisson_filt}

        # %% Pre-process video
        video_input, _ = preprocess_video(dir_video, Exp_ID, Params_pre, dir_network_input, \
            useSF=useSF, useTF=useTF, useSNR=useSNR, med_subtract=med_subtract, prealloc=prealloc) #

        # %% Determine active neurons in all frames using FISSA
        file_mask = dir_GTMasks + Exp_ID + '.mat' # foldr to save the temporal masks
        generate_masks(video_input, file_mask, list_thred_ratio, dir_parent, Exp_ID)
        del video_input

    # %% CNN training
    num_total = (list_nframes - list_TF_size).min() + 1 # number of frames used for CNN training. 
    if cross_validation == "use_all":
        list_CV = [nvideo]
    else: 
        list_CV = list(range(0,nvideo))
    for CV in list_CV:
        if cross_validation == "leave_one_out":
            list_Exp_ID_train = list_Exp_ID.copy()
            list_Exp_ID_val = [list_Exp_ID_train.pop(CV)]
        elif cross_validation == "train_1_test_rest":
            list_Exp_ID_val = list_Exp_ID.copy()
            list_Exp_ID_train = [list_Exp_ID_val.pop(CV)]
        elif cross_validation == "use_all":
            use_validation = False
            list_Exp_ID_train = list_Exp_ID.copy() 
        else:
            raise('wrong "cross_validation"')
        if not use_validation:
            list_Exp_ID_val = None # Afternatively, we can get rid of validation steps
        file_CNN = os.path.join(weights_path, 'Model_CV{}.h5'.format(CV))
        results = train_CNN(dir_network_input, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
            BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, Params_loss)

        # save training and validation loss after each eopch
        f = h5py.File(os.path.join(training_output_path, "training_output_CV{}.h5".format(CV)), "w")
        f.create_dataset("loss", data=results.history['loss'])
        f.create_dataset("dice_loss", data=results.history['dice_loss'])
        if use_validation:
            f.create_dataset("val_loss", data=results.history['val_loss'])
            f.create_dataset("val_dice_loss", data=results.history['val_dice_loss'])
        f.close()

    # # %% parameter optimization
    # parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, list_Mag, \
    #     list_Dimens, dir_network_input, weights_path, dir_GTMasks, dir_temp, dir_output, \
    #     batch_size_eval, useWT=useWT, useMP=useMP, load_exist=load_exist)
