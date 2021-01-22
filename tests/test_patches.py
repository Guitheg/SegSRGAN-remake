
from utils.visualisation import compare_n_img, visualiser
from utils.files import get_hr_seg_filepath_list
from utils.mri import MRI
from utils.mri_processing import get_tuple_lr_hr_seg_mri, lr_from_hr, read_mri, read_seg
from utils.patches import array_to_patches, create_mri_from_patches, create_patches_from_mri, make_a_patches_dataset, patches_to_array

from os.path import normpath, join


def runtest(config, *args, **kwargs):
    patchsize = (32, 32, 32)
    output_folder = "D:\\Projets\\srm4bmri\\outputs\\results"
    dataset_folder = "D:\\Projets\\srm4bmri\\dataset"
    hr_file_path = normpath(join(dataset_folder, "1010\\hr1010.nii.gz"))
    lr, hr, _ = lr_from_hr(hr_file_path, (2,2,2), 0.03, 1)
    lr_patch = array_to_patches(lr.get_img_array(), patchsize, 10)
    hr_patch = array_to_patches(hr.get_img_array(), patchsize, 10)
    visualiser(lr.get_img_array(), 1, 0.5)
    print(lr_patch.shape)
    compare_n_img(list(hr_patch), list(lr_patch), 1, 0.5)

