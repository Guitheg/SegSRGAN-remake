from main import DATA_EXAMPLE
from utils.mri_processing import lr_from_hr
from os.path import normpath, join

def runtest(config, *args, **kwargs):
    hr_file_path = normpath(join(DATA_EXAMPLE, "hr1010.nii.gz"))
    hr, lr, scaling_factor = lr_from_hr(hr_file_path, (1.8, 1.8, 1.8), 0.05, contrast_value=0.5)
    lr.save_mri(normpath(join(DATA_EXAMPLE, "lr1010.nii.gz")))
    
    create_patches()
    
    print(hr, lr)
    print(scaling_factor)