from configparser import ConfigParser
from os.path import join, normpath

from utils.mri_processing import lr_from_hr, read_seg
from utils.files import get_hr_seg_filepath_list
from utils.patches import create_patches_from_mri
import numpy as np
import pandas as pd


class MRI_Dataset():
    def __init__(self,
                 config : ConfigParser,
                 dataset_folder : str, 
                 mri_folder : str,
                 csv_listfile_path : str,
                 batchsize : int,
                 lr_resolution : tuple,
                 patchsize : int,
                 step : int,
                 percent_valmax : float,
                 *args, **kwargs):
        
        self.cfg = config
        self.dataset_folder = dataset_folder
        self.mri_folder = mri_folder
        self.csv_listfile_path = csv_listfile_path
        
        self.batchsize = batchsize
                
        self.lr_resolution = lr_resolution
        self.patchsize = patchsize
        self.step = step
        self.percent_valmax = percent_valmax
     
    def make_and_save_dataset_batchs(self, *args, **kwargs):
        train_batch_folder_name = self.cfg.get('Paths','Train_batch')
        val_batch_folder_name = self.cfg.get('Paths','Validatation_batch')
        test_batch_folder_name = self.cfg.get('Paths','Test_Batch')
        
        train_batch_folder_path = normpath(join(self.dataset_folder, train_batch_folder_name))
        val_batch_folder_path = normpath(join(self.dataset_folder, val_batch_folder_name))
        test_batch_folder_path = normpath(join(self.dataset_folder, test_batch_folder_name))
        
        train_fp_list, val_fp_list, test_fp_list = get_hr_seg_filepath_list(self.mri_folder, self.csv_listfile_path, self.cfg)
        
        self._save_data_base_batchs(train_fp_list, train_batch_folder_path)
        self._save_data_base_batchs(val_fp_list, val_batch_folder_path)
        self._save_data_base_batchs(test_fp_list, test_batch_folder_path)
     
    def _save_data_base_batchs(self, data_filespath_list : list, data_base_folder : str, *args, **kwargs):
        batch_index = 0
        remaining_patch = 0
        lr_gen_input_list = []
        hr_seg_dis_input_list = []
        
        for data_hr, data_seg in data_filespath_list:

            lr_img, hr_img, scaling_factor = lr_from_hr(data_hr, self.lr_resolution, self.percent_valmax)
            seg_img = read_seg(data_seg, scaling_factor)
            
            print(lr_img().shape)
            
            lr_gen_input, hr_seg_dis_input = create_patches_from_mri(lr_img, hr_img, seg_img, self.patchsize, self.step)
            
            lr_gen_input_list.append(lr_gen_input)
            hr_seg_dis_input_list.append(hr_seg_dis_input)
            
            buffer_lr = np.concatenate(np.asarray(lr_gen_input_list))
            buffer_hr = np.concatenate(np.asarray(hr_seg_dis_input_list))
    
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
        
        buffer_lr = np.concatenate(np.asarray(lr_gen_input_list))
        buffer_hr = np.concatenate(np.asarray(hr_seg_dis_input_list))

        buffer_lr = reshape_buffer(buffer_lr)
        buffer_hr = reshape_buffer(buffer_hr)
            
        lr_batch_name = f"{batch_index:04d}_batch_lr.npy"
        hr_seg_batch_name = f"{batch_index:04d}_batch_hr_seg.npy"
        
        np.save(normpath(join(data_base_folder, lr_batch_name), buffer_lr[:remaining_patch]))
        np.save(normpath(join(data_base_folder, hr_seg_batch_name), buffer_hr[:remaining_patch]))
        
        buffer_lr = buffer_lr[remaining_patch:]
        buffer_hr = buffer_hr[remaining_patch:]
        
        lr_gen_input_list = [buffer_lr]
        hr_seg_dis_input_list = [buffer_hr]
        
        return remaining_patch


def reshape_buffer(buffer):
    return buffer.reshape(-1, buffer.shape[-4], buffer.shape[-3], buffer.shape[-2], buffer.shape[-1])