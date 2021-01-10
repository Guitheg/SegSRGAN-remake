


from dataset.dataset_manager import MRI_Dataset


def runtest(config, *args, **kwargs):
    dataset_folder = "data_example\dataset_folder"
    mri_folder = "data_example"
    csv_listfile_path = "data_example\exemple.csv"
    
    dataset = MRI_Dataset(config, dataset_folder, mri_folder, csv_listfile_path, 10, (1,1,1), 16, 3, 0.5)
    dataset.make_and_save_dataset_batchs()