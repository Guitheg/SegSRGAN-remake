
from utils.mri import MRI
import numpy as np
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
    print(arr.shape)
    print(patch_shape, extraction_step)
    patches = extract_patches(arr, patch_shape, extraction_step)
    print(patches.shape)
    patches = patches.reshape(-1, patch_shape[0], patch_shape[1], patch_shape[2])
    # patches = patches.reshape(patches.shape[0], -1)
    if normalization is True:
        patches -= np.mean(patches, axis=0)
        patches /= np.std(patches, axis=0)
    print('%.2d patches have been extracted' % patches.shape[0])
    return patches


def create_patches_from_mri(lr : MRI, hr : MRI, seg : MRI, patchsize : tuple, stride : int, normalization : bool = False):
  
    # lr_patches_shape : (number_patches, patchsize[0], patchsize[1], patchsize[2])
    lr_patches = array_to_patches(lr(), patch_shape=patchsize, extraction_step=stride, normalization=normalization)
    hr_patches = array_to_patches(hr(), patch_shape=patchsize, extraction_step=stride, normalization=normalization)
    seg_patches = array_to_patches(seg(), patch_shape=patchsize, extraction_step=stride, normalization=normalization)
    
    # label_patches_shape : (number_patches, 2, patchsize[0], patchsize[1], patchsize[2])
    label_patches = np.swapaxes(np.stack((hr_patches, seg_patches)),  0, 1)
    
    return lr_patches, label_patches

def make_a_patches_dataset(mri_lr_hr_seg_list : list, patchsize : tuple, stride : int):
    dataset_lr_patches = []
    dataset_label_patches = []
    for lr, hr, seg in mri_lr_hr_seg_list:
        lr_patches, label_patches = create_patches_from_mri(lr, hr, seg, patchsize=patchsize, stride=stride)
        dataset_lr_patches.append(lr_patches)
        dataset_label_patches.append(label_patches)
        
    return dataset_lr_patches, dataset_label_patches