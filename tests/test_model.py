
from model.mri_srgan import MRI_SRGAN
from dataset.dataset_manager import MRI_Dataset
from tensorflow import keras

def runtest(config, *args, **kwargs):
    checkpoint_folder =  "D:\\Projets\\srm4bmri\\training_folder\\checkpoints"
    dataset_folder = "D:\\Projets\\srm4bmri\\dataset"
    logs_folder = "D:\\Projets\\srm4bmri\\outputs\\indices"
    batch_folder = "D:\\Projets\\srm4bmri\\training_folder\\batchs"
    csv_listfile_path = "D:\\Projets\\srm4bmri\\training_folder\\csv\\example.csv"
    shape = (16, 16, 16)
    # dataset = MRI_Dataset(config, 
    #                       batch_folder=batch_folder, 
    #                       mri_folder=dataset_folder,
    #                       csv_listfile_path=csv_listfile_path,
    #                       batchsize=128,
    #                       lr_downscale_factor=(2, 2, 2),
    #                       patchsize=shape,
    #                       step=2,
    #                       percent_valmax=0.05
    #                       )
    # dataset.make_and_save_dataset_batchs()

    mri_srgan = MRI_SRGAN("training", checkpoint_folder=checkpoint_folder, logs_folder=logs_folder)
    
    # mri_srgan.train(dataset, 2)
