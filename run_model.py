

import argparse
from configparser import ConfigParser
import sys
from utils.mri import MRI

import numpy as np
from model.segsrgan_model import SegSRGAN
from os.path import isfile, isdir, join
from utils.patches import array_to_patches, patches_to_array
from utils.mri_processing import read_mri
from utils.files import get_and_create_dir, get_environment

from main import CONFIG_INI_FILEPATH


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--running_name", "-n", help="the training name, for recovering", required=True)
    parser.add_argument("--filepath", "-f", help="mri path to apply the model", required=True)
    parser.add_argument("--output_dir", "-o", help="output folder path", required=True)
    parser.add_argument("--modelpath", "-m", help="model directory path", required=True)
    parser.add_argument("--patchsize", "-ps", help="tuple of the 3d patchsize. example : '16 16 16' ", required=True, nargs=3)
    parser.add_argument("--step", '-st', help="step/stride for patches construction", default=10)
    parser.add_argument('--percent_valmax', help="N trained on image on which we add gaussian noise with sigma equal to this % of val_max", default=0.03)
    
    args = parser.parse_args()
    
    config = ConfigParser()
    if not isfile(CONFIG_INI_FILEPATH):
        raise Exception("You must run 'build_env.py -f <home_folder>'")
    config.read(CONFIG_INI_FILEPATH)
    
    print(f"run_model.py -n {args.running_name} -f {args.filepath} -o {args.output_dir} -m {args.modelpath} -ps {args.patchsize} -st {args.step} ")

    home_folder = config.get('Path', 'Home')
    
    print(f"workspace : {home_folder}")
    
    try: 
        (home_folder, out_repo_path, training_repo_path, 
        dataset_repo_path, batch_repo_path, checkpoint_repo_path, 
        csv_repo_path, weights_repo_path, indices_repo_path, result_repo_path) = get_environment(home_folder, config)
    except Exception:
        raise Exception(f"Home folder has not been set. You must run 'build_env.py -f <home_folder>' script before launch the training")
    
    output_folder = get_and_create_dir(args.output_dir)
    name = args.running_name
    patchsize = (int(args.patchsize[0]), int(args.patchsize[1]), int(args.patchsize[2]))
    step = int(args.step)
    
    if isdir(args.modelpath):
        model_path = args.modelpath
    else:
        raise Exception(f"The path of {args.modelpath} is unknown or is not a folder")
    
    if isfile(args.filepath):
        mri_filepath = args.filepath
    else:
        raise Exception(f"The path of {args.filepath} is unknown")
    
    # MRI to patches
    mri = read_mri(mri_filepath)
    mri_patches = array_to_patches(mri(), patchsize, step)
    mri_patches = np.reshape(mri_patches, (-1, 1, patchsize[0], patchsize[1], patchsize[2]))
    np.save(join(output_folder, "mri_patches.npy"), mri_patches)
    # Load models
    segsrgan = SegSRGAN(name, model_path, shape=patchsize)
    
    sr_seg_patches = segsrgan.predict(mri_patches)
    
    print("Prediction ok")
    
    sr_patches = sr_seg_patches[:,0,:,:,:]
    seg_patches = sr_seg_patches[:,1,:,:,:]
    
    array_sr = patches_to_array(sr_patches, mri().shape, patchsize)
    array_seg = patches_to_array(seg_patches, mri().shape, patchsize)
    
    np.save(join(output_folder, "patches.npy"), sr_seg_patches)
    np.save(join(output_folder, "sr.npy"), array_sr)
    np.save(join(output_folder, "seg.npy"), array_seg)
    print(array_sr.shape)
    
    sr = MRI()
    sr.load_from_array(array_sr, (0.35000091791152954, 0.3472222089767456, 0.3472222089767456), mri.get_origin(), mri.get_direction())
    sr.save_mri(join(output_folder, "sr.nii.gz"))
    
    seg = MRI()
    seg.load_from_array(array_seg, (0.35000091791152954, 0.3472222089767456, 0.3472222089767456), mri.get_origin(), mri.get_direction())
    seg.save_mri(join(output_folder, "seg.nii.gz"))
    
    sys.exit(0)
    

if __name__ == "__main__":
    main()