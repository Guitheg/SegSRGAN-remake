from os.path import join
from utils.visualisation import compare_n_img, visualiser_n_img
import numpy as np

def runtest(config, *args, **kwargs):
    output_folder = "D:\\Projets\\srm4bmri\\outputs\\results"
    sr = np.load(join(output_folder, "patches.npy"))[:, 0, :, :, :]
    lr = np.load(join(output_folder, "mri_patches.npy"))[:, 0, :, :, :]
    list_patch_sr = list(sr)
    list_patch_lr = list(lr)

    compare_n_img(list_patch_lr, list_patch_sr, axis= 1, number=0.1)