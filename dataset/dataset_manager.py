from configparser import ConfigParser
from os.path import join, normpath
from typing import Union

from requests import patch

from utils.mri_processing import lr_from_hr, read_seg
from utils.files import get_and_create_dir, get_hr_seg_filepath_list
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
                 patchsize : Union[int, tuple],
                 step : int,
                 percent_valmax : float,
                 *args, **kwargs):
        
        self.cfg = config
        
        self.dataset_folder = dataset_folder
        self.mri_folder = mri_folder
        self.csv_listfile_path = csv_listfile_path
        
        self.batchsize = batchsize
                
        self.lr_resolution = lr_resolution
        if type(patchsize) == tuple:
            self.patchsize = patchsize
        else:
            self.patchsize = (patchsize, patchsize, patchsize)
        self.step = step
        self.percent_valmax = percent_valmax
        
        self.index = 0
        
        self.batchs_path_list = {self.cfg.get('Base_Header_Values','Train') : [],
                      self.cfg.get('Base_Header_Values','Validation') : [],
                      self.cfg.get('Base_Header_Values','Test') : []}
        
        self.initialize = False
     
    def make_and_save_dataset_batchs(self, *args, **kwargs):
        train_batch_folder_name = self.cfg.get('Paths','Train_batch')
        val_batch_folder_name = self.cfg.get('Paths','Validatation_batch')
        test_batch_folder_name = self.cfg.get('Paths','Test_Batch')
        
        train_batch_folder_path = get_and_create_dir(normpath(join(self.dataset_folder, train_batch_folder_name)))
        val_batch_folder_path = get_and_create_dir(normpath(join(self.dataset_folder, val_batch_folder_name)))
        test_batch_folder_path = get_and_create_dir(normpath(join(self.dataset_folder, test_batch_folder_name)))
        
        train_fp_list, val_fp_list, test_fp_list = get_hr_seg_filepath_list(self.mri_folder, self.csv_listfile_path, self.cfg)
        
        self._save_data_base_batchs(train_fp_list, train_batch_folder_path, base = self.cfg.get('Base_Header_Values','Train'))
        self._save_data_base_batchs(val_fp_list, val_batch_folder_path, base = self.cfg.get('Base_Header_Values','Validation'))
        self._save_data_base_batchs(test_fp_list, test_batch_folder_path, base = self.cfg.get('Base_Header_Values','Test'))
        
        self.initialize = True
    
    def __call__(self, base : str):
        if not self.initialize:
            raise Exception("Dataset has not been initialized")
        if self.batchs_path_list[base] == []:
            raise Exception(f"Dataset : {base} empty")
        for lr_path, hr_seg_path in self.batchs_path_list[base]:
            lr = np.load(lr_path)
            hr_seg = np.load(hr_seg_path)
            yield lr, hr_seg
    
    def __iter__(self, base : str):
        return self(base)
     
    def _save_data_base_batchs(self, data_filespath_list : list, data_base_folder : str, base : str, *args, **kwargs):
        batch_index = 0
        remaining_patch = 0
        lr_gen_input_list = []
        hr_seg_dis_input_list = []
        
        for data_hr, data_seg in data_filespath_list:

            lr_img, hr_img, scaling_factor = lr_from_hr(data_hr, self.lr_resolution, self.percent_valmax)
            seg_img = read_seg(data_seg, scaling_factor)
            
            lr_gen_input, hr_seg_dis_input = create_patches_from_mri(lr_img, hr_img, seg_img, self.patchsize, self.step)
            
            # shuffle lr_gen and hr_seg_dis here
            
            lr_gen_input_list.append(lr_gen_input)
            hr_seg_dis_input_list.append(hr_seg_dis_input)
            
            buffer_lr = np.concatenate(np.asarray(lr_gen_input_list))
            buffer_hr = np.concatenate(np.asarray(hr_seg_dis_input_list))
            buffer_lr = reshape_buffer(buffer_lr)
            buffer_hr = reshape_buffer(buffer_hr)
            
            while buffer_lr.shape[0] >= self.batchsize:
                
                lr_batch_name = f"{batch_index:04d}_batch_lr.npy"
                hr_seg_batch_name = f"{batch_index:04d}_batch_hr_seg.npy"
                lr_batch_path = normpath(join(data_base_folder, lr_batch_name))
                hr_seg_batch_path = normpath(join(data_base_folder, hr_seg_batch_name))
                
                np.save(lr_batch_path, buffer_lr[:self.batchsize])
                np.save(hr_seg_batch_path, buffer_hr[:self.batchsize])
                
                buffer_lr = buffer_lr[self.batchsize:]
                buffer_hr = buffer_hr[self.batchsize:]

                lr_gen_input_list = [buffer_lr]
                hr_seg_dis_input_list = [buffer_hr]
                
                batch_index += 1
                
                remaining_patch = buffer_lr.shape[0]
                
                self.batchs_path_list[base].append((lr_batch_path, hr_seg_batch_path))
        
        if remaining_patch > 0:
            
            buffer_lr = np.concatenate(np.asarray(lr_gen_input_list))
            buffer_hr = np.concatenate(np.asarray(hr_seg_dis_input_list))

            buffer_lr = reshape_buffer(buffer_lr)
            buffer_hr = reshape_buffer(buffer_hr)
                
            lr_batch_name = f"{batch_index:04d}_batch_lr.npy"
            hr_seg_batch_name = f"{batch_index:04d}_batch_hr_seg.npy"
            lr_batch_path = normpath(join(data_base_folder, lr_batch_name))
            hr_seg_batch_path = normpath(join(data_base_folder, hr_seg_batch_name))
                
            np.save(lr_batch_path, buffer_lr[:self.batchsize])
            np.save(hr_seg_batch_path, buffer_hr[:self.batchsize])
            
            buffer_lr = buffer_lr[remaining_patch:]
            buffer_hr = buffer_hr[remaining_patch:]
            
            lr_gen_input_list = [buffer_lr]
            hr_seg_dis_input_list = [buffer_hr]
            
            self.batchs_path_list[base].append((lr_batch_path, hr_seg_batch_path))
        
        # shuffle datas here
        
        return remaining_patch


def reshape_buffer(buffer):
    return buffer.reshape(-1, buffer.shape[-4], buffer.shape[-3], buffer.shape[-2], buffer.shape[-1])