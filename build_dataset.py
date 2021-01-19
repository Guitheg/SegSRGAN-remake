from configparser import ConfigParser
from pkgutil import iter_modules
import sys
from model.mri_srgan import MRI_SRGAN
from dataset.dataset_manager import MRI_Dataset
from utils.files import get_and_create_dir, get_environment
from main import CONFIG_INI_FILEPATH
from os.path import normpath, join, isfile
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-n", help="the dataset name...", required=True)
    parser.add_argument("--csv_name", "-csv", help="file path of the csv listing mri path", required=True)
    parser.add_argument("--batchsize", "-bs", help="batchsize of the training", default=32)
    parser.add_argument("--downscale_factor", "-lr", help="factor for downscaling hr image by. it's a tuple of 3. example : 0.5 0.5 0.5", nargs=3, default=(2,2,2))
    parser.add_argument("--patchsize", "-ps", help="tuple of the 3d patchsize. example : '16 16 16' ", nargs=3, default=(32, 32, 32))
    parser.add_argument("--step", '-st', help="step/stride for patches construction", default=4)
    parser.add_argument('--percent_valmax', help="N trained on image on which we add gaussian noise with sigma equal to this % of val_max", default=0.03)
    parser.add_argument('--save_lr', help="if you want to save lr mri", action="store_true")
    parser.add_argument('--merge_hr_seg', help="if you want to marge hr and segmentation for label", action="store_true")

    args = parser.parse_args()
    
    config = ConfigParser()
    if not isfile(CONFIG_INI_FILEPATH):
        raise Exception("You must run 'build_env.py -f <home_folder>'")
    config.read(CONFIG_INI_FILEPATH)
    
    print(f"build_dataset.py -n {args.dataset_name} -csv {args.csv_name} -bs {args.batchsize} -lr {args.downscale_factor} -ps {args.patchsize} -st {args.step} --percent_valmax {args.percent_valmax}")

    home_folder = config.get('Path', 'Home')
    
    print(f"workspace : {home_folder}")
    
    try: 
        (home_folder, out_repo_path, training_repo_path, 
        dataset_repo_path, batch_repo_path, checkpoint_repo_path, 
        csv_repo_path, weights_repo_path, indices_repo_path, result_repo_path) = get_environment(home_folder, config)
        
    except Exception:
        
        raise Exception(f"Home folder has not been set. You must run 'build_env.py -f <home_folder>' script before launch the training")
    
    csv_listfile_path = normpath(join(csv_repo_path, args.csv_name))
    
    if not isfile(csv_listfile_path):
        raise Exception(f"{csv_listfile_path} unknown. you must put {args.csv_name} in {csv_repo_path} folder")
    
    dataset_name = args.dataset_name
    batchsize = int(args.batchsize)
    patchsize = (int(args.patchsize[0]), int(args.patchsize[1]), int(args.patchsize[2]))
    lr_downscale_factor = (float(args.downscale_factor[0]), float(args.downscale_factor[1]), float(args.downscale_factor[2]))
    step = int(args.step)
    percent_valmax = float(args.percent_valmax)
    
    print("Dataset creation : preprocess and patches generation...")
    
    batch_repo_path = get_and_create_dir(join(batch_repo_path, dataset_name))
    
    dataset = MRI_Dataset(config, batch_folder = batch_repo_path)
    dataset.make_and_save_dataset_batchs(mri_folder = dataset_repo_path,
                                         csv_listfile_path = csv_listfile_path,
                                         batchsize=batchsize,
                                         lr_downscale_factor=lr_downscale_factor,
                                         patchsize=patchsize,
                                         step=step,
                                         percent_valmax=percent_valmax,
                                         save_lr = args.save_lr,
                                         merge_hr_seg=args.merge_hr_seg)
    
    print(f"Done !")
    print(f"Dataset create at : {batch_repo_path} with :\n*batchsize of : {batchsize}")
    print(f"*patchsize of : {patchsize} by {step} step\n*gaussian noise of : {percent_valmax}\n*downscale factor of : {lr_downscale_factor}")
    sys.exit(0)
    
if __name__ == "__main__":
    main()