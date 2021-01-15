
from utils.files import get_hr_seg_filepath_list
from utils.mri import MRI
from main import DATA_EXAMPLE
from utils.mri_processing import get_tuple_lr_hr_seg_mri, lr_from_hr, read_mri, read_seg
from utils.patches import array_to_patches, create_patches_from_mri, make_a_patches_dataset

from os.path import normpath, join


def runtest(config, *args, **kwargs):
    tr, va, te = get_hr_seg_filepath_list(DATA_EXAMPLE, normpath(join(DATA_EXAMPLE, "exemple.csv")), config)
    lr_hr_seg_mri_list = []
    for hr_seg_filepath in tr:
        lr_hr_seg_mri_list.append(get_tuple_lr_hr_seg_mri(hr_seg_filepath, (2,2,2), 0.03))
    
    dataset_lr_patches, dataset_label_patches = make_a_patches_dataset(lr_hr_seg_mri_list, (8, 9, 10), 2)
