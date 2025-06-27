import numpy as np
import cv2
from cv2 import dft as fftn
from cv2 import idft as ifftn
from past.utils import old_div
from numpy.fft import ifftshift
import os
import tifffile as tiff

def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", shifts_lb=None, shifts_ub=None, max_shifts=(10, 10),
                         use_cuda=False):
    """

    adapted from SIMA (https://github.com/losonczylab) and the
    scikit-image (http://scikit-image.org/) package.


    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Args:
        src_image : ndarray
            Reference image.

        target_image : ndarray
            Image to register.  Must be same dimensionality as ``src_image``.

        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel.  Default is 1 (no upsampling)

        space : string, one of "real" or "fourier"
            Defines how the algorithm interprets input data.  "real" means data
            will be FFT'd to compute the correlation, while "fourier" data will
            bypass FFT of input data.  Case insensitive.

        use_cuda : bool, optional
            Use skcuda.fft (if available). Default: False

    Returns:
        shifts : ndarray
            Shift vector (in pixels) required to register ``target_image`` with
            ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

        error : float
            Translation invariant normalized RMS error between ``src_image`` and
            ``target_image``.

        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).

    Raises:
     NotImplementedError "Error: register_translation only supports "
                                  "subpixel registration for 2D images"

     ValueError "Error: images must really be same size for "
                         "register_translation"

     ValueError "Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument."

    References:
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must really be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # if use_cuda:
    #     from skcuda.fft import Plan
    #     from skcuda.fft import fft as cudafft
    #     from skcuda.fft import ifft as cudaifft
        # try:
        #     cudactx # type: ignore
        # except NameError:
        #     init_cuda_process()

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        
        src_freq_1 = fftn(
            src_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        src_freq = src_freq_1[:, :, 0] + 1j * src_freq_1[:, :, 1]
        src_freq = np.array(src_freq, dtype=np.complex128, copy=False)
        target_freq_1 = fftn(
            target_image, flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)
        target_freq = target_freq_1[:, :, 0] + 1j * target_freq_1[:, :, 1]
        target_freq = np.array(
            target_freq, dtype=np.complex128, copy=False)

    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()

    image_product_cv = np.dstack(
        [np.real(image_product), np.imag(image_product)])
    cross_correlation = fftn(
        image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
    cross_correlation = cross_correlation[:,
                                        :, 0] + 1j * cross_correlation[:, :, 1]

    # Locate maximum
    new_cross_corr = np.abs(cross_correlation)

    if (shifts_lb is not None) or (shifts_ub is not None):

        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0]:shifts_lb[0], :] = 0
        else:
            new_cross_corr[:shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0]:, :] = 0

        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1]:shifts_lb[1]] = 0
        else:
            new_cross_corr[:, :shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1]:] = 0
    else:

        new_cross_corr[max_shifts[0]:-max_shifts[0], :] = 0

        new_cross_corr[:, max_shifts[1]:-max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(old_div(axis_size, 2))
                          for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:

        src_amp = old_div(np.sum(np.abs(src_freq) ** 2), src_freq.size)
        target_amp = old_div(
            np.sum(np.abs(target_freq) ** 2), target_freq.size)
        CCmax = cross_correlation.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = old_div(np.round(shifts * upsample_factor), upsample_factor)
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(old_div(upsampled_region_size, 2.0))
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor

        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + old_div(maxima, upsample_factor)
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, src_freq, _compute_phasediff(CCmax)

def apply_shift_iteration(img, shift, border_nan:bool=False, border_type=cv2.BORDER_REFLECT):
    # todo todocument

    sh_x_n, sh_y_n = shift
    w_i, h_i = img.shape
    M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
    min_, max_ = np.nanmin(img), np.nanmax(img)
    img = np.clip(cv2.warpAffine(img, M, (h_i, w_i),
                                 flags=cv2.INTER_CUBIC, borderMode=border_type), min_, max_)
    if border_nan is not False:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
        max_h, max_w = np.ceil(np.maximum(
            (max_h, max_w), shift)).astype(int)
        min_h, min_w = np.floor(np.minimum(
            (min_h, min_w), shift)).astype(int)
        if border_nan is True:
            img[:max_h, :] = np.nan
            if min_h < 0:
                img[min_h:, :] = np.nan
            img[:, :max_w] = np.nan
            if min_w < 0:
                img[:, min_w:] = np.nan
        elif border_nan == 'min':
            img[:max_h, :] = min_
            if min_h < 0:
                img[min_h:, :] = min_
            img[:, :max_w] = min_
            if min_w < 0:
                img[:, min_w:] = min_
        elif border_nan == 'copy':
            if max_h > 0:
                img[:max_h] = img[max_h]
            if min_h < 0:
                img[min_h:] = img[min_h-1]
            if max_w > 0:
                img[:, :max_w] = img[:, max_w, np.newaxis]
            if min_w < 0:
                img[:, min_w:] = img[:, min_w-1, np.newaxis]

    return img

def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None):
    """
    adapted from SIMA (https://github.com/losonczylab) and the scikit-image (http://scikit-image.org/) package.

    Unless otherwise specified by LICENSE.txt files in individual
    directories, all code is

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Args:
        data : 2D ndarray
            The input data array (DFT of original data) to upsample.

        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.

        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.

        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)

    Returns:
        output : 2D ndarray
                The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor)) *
        (ifftshift(np.arange(data.shape[1]))[:, None] -
         np.floor(old_div(data.shape[1], 2))).dot(
             np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1])
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor)) *
        (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            ifftshift(np.arange(data.shape[0]))[None, :] -
            np.floor(old_div(data.shape[0], 2)))
    )

    if data.ndim > 2:
        pln_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[2] * upsample_factor)) *
        (np.arange(upsampled_region_size[2])[:, None] - axis_offsets[2]).dot(
                ifftshift(np.arange(data.shape[2]))[None, :] -
                np.floor(old_div(data.shape[2], 2))))

    # output = np.tensordot(np.tensordot(row_kernel,data,axes=[1,0]),col_kernel,axes=[1,0])
    output = np.tensordot(row_kernel, data, axes = [1,0])
    output = np.tensordot(output, col_kernel, axes = [1,0])

    if data.ndim > 2:
        output = np.tensordot(output, pln_kernel, axes = [1,1])
    #output = row_kernel.dot(data).dot(col_kernel)
    return output

def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be zero if images are non-negative).

    Args:
        cross_correlation_max : complex
            The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)

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
    num_windows = int(old_div(T, window))
    num_frames = num_windows * window
    if exclude_nans:
        img = np.nanmedian(np.nanmean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)
    else:
        img = np.median(np.mean(np.reshape(
            mat[:num_frames], (window, num_windows, d1, d2)), axis=0), axis=0)

    return img


if __name__ == '__main__':
    
    dwonsampling = 4
    # load from disk
    for directory_id in range(1,2):
        directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/mini2p/ca1/' + str(directory_id+1) + '/'
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/2p_fiberscopy/' + str(directory_id+1) + '/'
        # directory_path = '/mnt/nas/YZ_personal_storage/Private/MC/2p_benchtop/2p_148d/trial_2p_' + str(directory_id+1) + '/motion/'
        all_files = os.listdir(directory_path)
    
        tiff_files = [filename for filename in all_files if filename.endswith('.tiff')]

        save_path = directory_path + 'Self_Rigid_result/'

        os.makedirs(save_path, exist_ok=True)

        for fnames in tiff_files:
            fnames = directory_path + fnames
            session_name = fnames.split('/')[-1].split('.')[0]
            src_video = tiff.imread(fnames)
            t, h, w = src_video.shape

            target = bin_median(src_video)
            target = cv2.resize(target, (h // dwonsampling, w // dwonsampling), interpolation=cv2.INTER_CUBIC)

            new_video = []

            for t in range(t):
                img_old = np.float64(src_video[t])
                img2rigid = cv2.resize(img_old, (h // dwonsampling, w // dwonsampling), interpolation=cv2.INTER_CUBIC)

                shifts, src_freq, phasediff = register_translation(img2rigid, target, 10)

                shifts = dwonsampling * shifts

                img_rigid = apply_shift_iteration(img_old, (-shifts[0], -shifts[1]))

                new_video.append(img_rigid)

            new_video = np.stack(new_video).astype('int32')

            output_file = os.path.join(save_path, '{}_reg.tif'.format(session_name))
            tiff.imwrite(output_file, new_video)

        print(directory_id)
