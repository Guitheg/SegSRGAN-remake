


from dataset.dataset_manager import MRI_Dataset


def runtest(config, *args, **kwargs):
    dataset_folder = "data_example\dataset_folder"
    mri_folder = "data_example"
    csv_listfile_path = "data_example\exemple.csv"
    
    dataset = MRI_Dataset(config, dataset_folder, mri_folder, csv_listfile_path, batchsize=128, 
                          lr_resolution=(1,1,1), patchsize=(16,16,16), step=4, percent_valmax=0.5)
    dataset.make_and_save_dataset_batchs()
    
    for lr, hr in dataset('Train'):
        print(lr.shape, hr.shape)