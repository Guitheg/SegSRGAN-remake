

import argparse
from configparser import ConfigParser
from utils.patches import test_by_patch
from model.mri_srgan import MRI_SRGAN
import sys
from utils.mri import MRI

import numpy as np
from os.path import isfile, isdir, join, basename
from utils.mri_processing import read_mri
from utils.files import get_and_create_dir, get_environment

from main import CONFIG_INI_FILEPATH

# python run_model.py -n test_running -f D:\\Projets\\srm4bmri\\dataset\\1010\\hr1010.nii.gz -m train_mri_srgan

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--running_name", "-n", help="the training name, for recovering", required=True)
    parser.add_argument("--filepath", "-f", help="mri path to apply the model", required=True)
    parser.add_argument("--modelname", "-m", help="model name", required=True)
    parser.add_argument('--percent_valmax', help="N trained on image on which we add gaussian noise with sigma equal to this % of val_max", default=0.03)
    
    args = parser.parse_args()
    
    config = ConfigParser()
    if not isfile(CONFIG_INI_FILEPATH):
        raise Exception("You must run 'build_env.py -f <home_folder>'")
    config.read(CONFIG_INI_FILEPATH)
    
    print(f"run_model.py -n {args.running_name} -f {args.filepath} -m {args.modelname}")

    home_folder = config.get('Path', 'Home')
    
    print(f"workspace : {home_folder}")
    
    try: 
        (home_folder, out_repo_path, training_repo_path, 
        dataset_repo_path, batch_repo_path, checkpoint_repo_path, 
        csv_repo_path, weights_repo_path, indices_repo_path, result_repo_path) = get_environment(home_folder, config)
    except Exception:
        raise Exception(f"Home folder has not been set. You must run 'build_env.py -f <home_folder>' script before launch the training")
    
    name = args.running_name
    patchsize = (32, 32, 32)
    step = 4
    
    if isdir(join(weights_repo_path, args.modelname)):
        modelname = args.modelname
    else:
        raise Exception(f"The path of {args.modelname} is unknown or is not the name of a model in the folder : {weights_repo_path}")
    
    if isfile(args.filepath):
        mri_filepath = args.filepath
    else:
        raise Exception(f"The path of {args.filepath} is unknown")
    
    mri = read_mri(mri_filepath)
    
    mri_srgan = MRI_SRGAN(name = modelname, 
                          checkpoint_folder = checkpoint_repo_path,
                          weight_folder = weights_repo_path,
                          logs_folder = indices_repo_path
                          )
    mri_srgan.load_weights()
    
    sr_mri = test_by_patch(mri, mri_srgan)
    sr_mri.save_mri(join(out_repo_path, "SR_"+basename(mri.filepath)))
    
    sys.exit(0)
    
if __name__ == "__main__":
    main()