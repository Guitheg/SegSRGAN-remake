
from utils.mri_processing import lr_from_hr
from os.path import normpath, join

def runtest(config, *args, **kwargs):
    output_folder = "D:\\Projets\\srm4bmri\\outputs\\results"
    dataset_folder = "D:\\Projets\\srm4bmri\\dataset"
    hr_file_path = normpath(join(dataset_folder, "1010\\hr1010.nii.gz"))
    hr, lr, scaling_factor = lr_from_hr(hr_file_path, (0.5, 0.5, 0.5), 0.03, contrast_value=0.5)
    lr.save_mri(normpath(join(output_folder, "lr1010.nii.gz")))
    
    print(hr, lr)
    print(scaling_factor)