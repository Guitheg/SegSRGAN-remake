from configparser import ConfigParser
from os.path import join, normpath

from utils.mri_processing import create_patches, lr_from_hr, read_seg
from utils.files import get_datas_filespath
import numpy as np
import pandas as pd


class MRI_Dataset():
    def __init__(self,
                 config : ConfigParser,
                 mri_folder : str,
                 csv_listfile_path : str,
                 batchsize : int,
                 lr_resolution : tuple,
                 patchsize : int,
                 step : int,
                 percent_valmax : float,
                 *args, **kwargs):
        
        self.cfg = config
        self.mri_folder = mri_folder
        self.csv_listfile_path = csv_listfile_path
        
        self.batchsize = batchsize
                
        self.lr_resolution = lr_resolution
        self.patchsize = patchsize
        self.step = step
        self.percent_valmax = percent_valmax
        
        
     
    def make_and_save_dataset_batchs(self, dataset_folder : str, *args, **kwargs):
        train_batch_folder_name = self.cfg.get('Path','Train_batch')
        val_batch_folder_name = self.cfg.get('Path','Validatation_batch')
        test_batch_folder_name = self.cfg.get('Path','Test_Batch')
        
        train_batch_folder_path = normpath(join(dataset_folder, train_batch_folder_name))
        val_batch_folder_path = normpath(join(dataset_folder, val_batch_folder_name))
        test_batch_folder_path = normpath(join(dataset_folder, test_batch_folder_name))
        
        train_fp_df, val_fp_df, test_fp_df = get_datas_filespath(self.mri_folder, self.csv_listfile_path, self.cfg)
        
        self._save_data_base_batchs(train_fp_df, train_batch_folder_path)
        self._save_data_base_batchs(val_fp_df, val_batch_folder_path)
        self._save_data_base_batchs(test_fp_df, test_batch_folder_path)
     
    def _save_data_base_batchs(self, data_filespath_df : pd.DataFrame, data_base_folder : str, *args, **kwargs):
        hr_header = self.cfg.get('CSV_Header','HR_Header')
        seg_header = self.cfg.get('CSV_Header','Seg_Header')
        
        batch_index = 0
        remaining_patch = 0
        lr_gen_input_list = []
        hr_seg_dis_input_list = []
        
        for _, row in data_filespath_df.iterrows():
            
            lr_img, hr_img, scaling_factor = lr_from_hr(row[hr_header], self.lr_resolution, self.percent_valmax)
            seg_img = read_seg(row[seg_header], scaling_factor)
            
            lr_gen_input, hr_seg_dis_input = create_patches(lr_img, hr_img, seg_img, self.patchsize, self.step)
            
            lr_gen_input_list.append(lr_gen_input)
            hr_seg_dis_input_list.append(hr_seg_dis_input)
            
            buffer_lr = np.concatenate(np.asarray(lr_gen_input_list))
            buffer_hr = np.concatenate(np.asarray(hr_seg_dis_input))
    
            buffer_lr = reshape_buffer(buffer_lr)
            buffer_hr = reshape_buffer(buffer_hr)
            
            while buffer_lr.shape[0] >= self.batchsize:
                
                lr_batch_name = f"{batch_index:04d}_batch_lr.npy"
                hr_seg_batch_name = f"{batch_index:04d}_batch_hr_seg.npy"
                
                np.save(normpath(join(data_base_folder, lr_batch_name), buffer_lr[:self.batchsize]))
                np.save(normpath(join(data_base_folder, hr_seg_batch_name), buffer_hr[:self.batchsize]))
                
                buffer_lr = buffer_lr[self.batchsize:]
                buffer_hr = buffer_hr[self.batchsize:]
                
                lr_gen_input_list = [buffer_lr]
                hr_seg_dis_input_list = [buffer_hr]
                
                batch_index += 1
                
                remaining_patch = buffer_lr.shape[0]
            
        return remaining_patch
    
def reshape_buffer(buffer):
    return buffer.reshape(-1, buffer.shape[-4], buffer.shape[-3], buffer.shape[-2], buffer.shape[-1])