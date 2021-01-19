from typing import Union
import numpy as np
import SimpleITK as sitk
from utils.mri import MRI
from scipy import ndimage
import math

def read_mri(mri_filepath):
    mri = MRI()
    mri.load_from_file(mri_filepath)
    return mri

def modcrop3D(img, modulo):
    """To avoid lr final shape different from hr shape or seg shape (after the downsampling and the interpolation)
    Args:
        img (ndarray): [image array]
        modulo (tuple): [scaling factor]

    Returns:
        [ndarray]: [reshaped image array]
    """
    img = img[0:int(img.shape[0] - math.fmod(img.shape[0], modulo[0])), 
              0:int(img.shape[1] - math.fmod(img.shape[1], modulo[1])), 
              0:int(img.shape[2] - math.fmod(img.shape[2], modulo[2]))]
    return img

def get_scaling_factor_from_raw_resolution(img_ref_path : str, res : tuple):
    ref_img = read_mri(img_ref_path)
    scaling_factor = [a/b for a, b in zip(ref_img.get_resolution(), res)]
    return scaling_factor
    
def get_tuple_lr_hr_seg_mri(hr_seg_filepath : tuple, 
                            scaling_factor : tuple, 
                            percent_valmax : float, 
                            contrast_value : float = 1.0, 
                            order : int = 3):
    hr_file_path, seg_file_path = hr_seg_filepath
    lr, hr, _ = lr_from_hr(hr_file_path, scaling_factor, percent_valmax, contrast_value, order)
    seg = read_seg(seg_file_path, scaling_factor)
    return lr, hr, seg
    
def lr_from_hr(hr_file_path : str, 
               scaling_factor : tuple,
               percent_valmax : float,
               contrast_value : float = 1.0,
               order : int = 3):
    
    hr = read_mri(hr_file_path)
    hr(modcrop3D(hr(), scaling_factor))
    
    lr_res = [a*b for a, b in zip(hr.get_resolution(), scaling_factor)]
    
    # Downsample parameters
    sigma_blur = lr_res / (2 * np.sqrt(2*np.log(2)) )
    blur_img = ndimage.filters.gaussian_filter(hr(), sigma = sigma_blur)
    
    # Downsample hr image to lr_image
    scales = [ 1 / float(idx_scale) for idx_scale in scaling_factor]
    lr_array = ndimage.zoom(blur_img, zoom = scales, order = 0)
    
    # Contraste changes
    lr_array = change_contraste(lr_array, contrast_value)
    hr_array = change_contraste(hr(), contrast_value) 
    hr(hr_array)
    
    # Adding noises
    lr_array = add_noise(lr_array, percent_valmax)
    
    # Normalization
    lr_array, hr_array = max_normalisation(lr_array, hr())
    hr(hr_array)
    
    # Interpolate lr image to have the same shape (resolution) as the hr image
    lr_array = ndimage.zoom(lr_array, zoom=scaling_factor, order=order)

    lr = MRI()
    lr.load_from_array(lr_array, lr_res, hr.get_origin(), hr.get_direction())

    if lr().shape != hr().shape:
        raise Exception(f"lr shape and hr shape are different : lr{lr().shape}, hr{hr().shape}")

    return lr, hr, scaling_factor

def read_seg(seg_file_path : str, scaling_factor : tuple):
    seg = read_mri(seg_file_path)
    seg(modcrop3D(seg(), scaling_factor))
    return seg
    
def change_contraste(img, power):
    return img**power

def add_noise(lr, percent_valmax):
    sigma = percent_valmax * np.max(lr)
    lr += np.random.normal(scale = sigma, size = lr.shape)
    lr[lr < 0] = 0
    return lr
    
def max_normalisation(lr, hr):
    max_value = np.max(lr)
    norm_lr = lr / max_value
    norm_hr = hr / max_value
    return norm_lr, norm_hr

