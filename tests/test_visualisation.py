from os.path import join
from utils.visualisation import visualiser_n_img
import numpy as np

def runtest(config, *args, **kwargs):
    output_folder = "D:\\Projets\\srm4bmri\\outputs\\results"
    sr = np.load(join(output_folder, "mri_patches.npy"))[:, 0, :, :, :]
    list_patch = list(sr)
    print([i.shape for i in list_patch])
    visualiser_n_img(list_patch, axis= 1, number=0.5)