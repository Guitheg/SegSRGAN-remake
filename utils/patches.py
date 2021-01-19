
from utils.mri import MRI
import numpy as np
from itertools import product
from sklearn.feature_extraction.image import extract_patches


def array_to_patches(arr, patch_shape=(3, 3, 3), extraction_step=1, normalization=False):
    # from SegSRGAN author : koopa31
    # Make use of skleanr function extract_patches
    # https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/feature_extraction/image.py
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content.
    Parameters
    ----------
    arr : 3darray
      3-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
      Indicates the shape of the patches to be extracted. If an
      integer is given, the shape will be a hypercube of
      sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
      Indicates step size at which extraction shall be performed.
      If integer is given, then the step is uniform in all dimensions.
    normalization : bool
        Enable normalization of the patches
    Returns
    -------
    patches : strided ndarray
      2n-dimensional array indexing patches on first n dimensions and
      containing patches on the last n dimensions. These dimensions
      are fake, but this way no data is copied. A simple reshape invokes
      a copying operation to obtain a list of patches:
      result.reshape([-1] + list(patch_shape))
    """
    patches = extract_patches(arr, patch_shape, extraction_step)
    patches = patches.reshape(-1, patch_shape[0], patch_shape[1], patch_shape[2])
    # patches = patches.reshape(patches.shape[0], -1)
    if normalization is True:
        patches -= np.mean(patches, axis=0)
        patches /= np.std(patches, axis=0)
    print('%.2d patches have been extracted' % patches.shape[0])
    return patches

def patches_to_array(patches, array_shape, patch_shape=(3, 3, 3)):
    """
    Swicth from the patches to the image
    :param patches: patches array
    :param array_shape: shape of the array
    :param patch_shape: shape of the patches
    :return: array
    """
    # Adapted from 2D reconstruction from sklearn
    # https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/feature_extraction/image.py
    # SyntaxError: non-default argument follows default argument : exchange "array_shape" and "patch_shape"
    patches = patches.reshape(len(patches),*patch_shape)
    i_x, i_y, i_z = array_shape
    p_x, p_y, p_z = patch_shape
    array = np.zeros(array_shape)
    # compute the dimensions of the patches array
    n_x = i_x - p_x + 1
    n_y = i_y - p_y + 1
    n_z = i_z - p_z + 1
    for p, (i, j, k) in zip(patches, product(range(n_x), range(n_y), range(n_z))):
        array[i:i + p_x, j:j + p_y, k:k + p_z] += p
  
    for (i, j, k) in product(range(i_x), range(i_y), range(i_z)):
        array[i, j, k] /= float(min(i + 1, p_x, i_x - i) * min(j + 1, p_y, i_y - j) * min(k + 1, p_z, i_z - k))
    return array
  
def create_patches_from_mri(lr : MRI, hr : MRI, seg : MRI, patchsize : tuple, stride : int, normalization : bool = False, merge_hr_seg : bool = True):
  
    # lr_patches_shape : (number_patches, 1, patchsize[0], patchsize[1], patchsize[2])
    lr_patches = array_to_patches(lr(), patch_shape=patchsize, extraction_step=stride, normalization=normalization)
    lr_patches = np.reshape(lr_patches, (-1, 1, patchsize[0], patchsize[1], patchsize[2]))
    hr_patches = array_to_patches(hr(), patch_shape=patchsize, extraction_step=stride, normalization=normalization)
    seg_patches = array_to_patches(seg(), patch_shape=patchsize, extraction_step=stride, normalization=normalization)
    
    if merge_hr_seg:
        # label_patches_shape : (number_patches, 2, patchsize[0], patchsize[1], patchsize[2])
        label_patches = concatenante_hr_seg(hr_patches, seg_patches)
    else:
        # label_patches_shape : (number_patches, 1, patchsize[0], patchsize[1], patchsize[2])
        label_patches = np.reshape(hr_patches, (-1, 1, patchsize[0], patchsize[1], patchsize[2]))
    
    return lr_patches, label_patches

def concatenante_hr_seg(hr_patches, seg_patches):
    label_patches = np.swapaxes(np.stack((hr_patches, seg_patches)),  0, 1)
    return label_patches

def make_a_patches_dataset(mri_lr_hr_seg_list : list, patchsize : tuple, stride : int):
    dataset_lr_patches = []
    dataset_label_patches = []
    for lr, hr, seg in mri_lr_hr_seg_list:
        lr_patches, label_patches = create_patches_from_mri(lr, hr, seg, patchsize=patchsize, stride=stride)
        dataset_lr_patches.append(lr_patches)
        dataset_label_patches.append(label_patches)
        
    return dataset_lr_patches, dataset_label_patches