from configparser import ConfigParser
from pkgutil import iter_modules
from model.mri_srgan import MRI_SRGAN
from dataset.dataset_manager import MRI_Dataset
from utils.files import get_environment
from main import CONFIG_INI_FILEPATH
from os.path import normpath, join, isfile, isdir
import argparse

def main():
    print([i.name for i in iter_modules()])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_name", "-n", help="the training name, for recovering", required=True)
    parser.add_argument("--dataset_name", "-d", help="name of the dataset", required=True)
    parser.add_argument('--n_epochs','-e', help="number of epochs", default=1)
    parser.add_argument('--mri_to_test', '-t', help="mri to test")
    
    args = parser.parse_args()
    
    config = ConfigParser()
    if not isfile(CONFIG_INI_FILEPATH):
        raise Exception("You must run 'build_env.py -f <home_folder>'")
    config.read(CONFIG_INI_FILEPATH)
    
    print(f"train.py -n {args.training_name} -d {args.dataset_name} -e {args.n_epochs} -t {args.mri_to_test}")

    home_folder = config.get('Path', 'Home')
    
    print(f"workspace : {home_folder}")
    
    try: 
        (home_folder, out_repo_path, training_repo_path, 
        dataset_repo_path, batch_repo_path, checkpoint_repo_path, 
        csv_repo_path, weights_repo_path, indices_repo_path, result_repo_path) = get_environment(home_folder, config)
    except Exception:
        raise Exception(f"Home folder has not been set. You must run 'build_env.py -f <home_folder>' script before launch the training")
    
    if args.mri_to_test:
        if not isfile(args.mri_to_test):
            raise Exception(f'{args.mri_to_test} is unknown, is not a file')
        else:
            mri_to_test = args.mri_to_test
    else:
        mri_to_test = None
    
    training_name = args.training_name
    n_epochs = int(args.n_epochs)
    batch_repo_path = join(batch_repo_path, args.dataset_name)
    
    if not isdir(batch_repo_path):
        raise Exception(f"{batch_repo_path} is unknown. Your must run 'build_dataset.py -n <dataset_name> -csv <csvlistfile_path>' to create the wanted dataset")
    
    print(f"Loading dataset from {batch_repo_path}")
    
    dataset = MRI_Dataset(config, batch_repo_path)
    dataset.load_dataset()
    
    print(f"Training creation : {training_name}")
    print(f"Checkpoint folder at : {join(checkpoint_repo_path, training_name)}")
    print(f"Logs_folder at : {join(indices_repo_path, training_name)}")
    segsrgan_trainer = MRI_SRGAN(name = training_name,
                                 checkpoint_folder=checkpoint_repo_path,
                                 weight_folder=weights_repo_path,
                                 logs_folder=indices_repo_path)
    
    print("Training...")
    
    segsrgan_trainer.train(dataset, n_epochs=n_epochs, mri_to_visualize=mri_to_test, output_dir=out_repo_path)
    
if __name__ == "__main__":
    main()