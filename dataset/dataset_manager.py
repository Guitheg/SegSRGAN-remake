from configparser import ConfigParser
import os
from os.path import join, normpath, basename
from typing import Union

from requests import patch

from utils.mri_processing import lr_from_hr, read_seg
from utils.files import get_and_create_dir, get_hr_seg_filepath_list
from utils.patches import create_patches_from_mri
import numpy as np
import pandas as pd

LR_BATCH = "_batch_lr.npy"
LABEL_BATCH = "_batch_label.npy"

def reshape_buffer(buffer):
    return buffer.reshape(-1, buffer.shape[-4], buffer.shape[-3], buffer.shape[-2], buffer.shape[-1])

class MRI_Dataset():
    
    def __init__(self,
                 config : ConfigParser,
                 batch_folder : str, 
                 *args, **kwargs):
        
        self.cfg = config
        
        self.batch_folder = batch_folder
        self.train_batch_folder_name = self.cfg.get('Batch_Path','Train_batch')
        self.val_batch_folder_name = self.cfg.get('Batch_Path','Validatation_batch')
        self.test_batch_folder_name = self.cfg.get('Batch_Path','Test_Batch')
        
        self.train_batch_folder_path = get_and_create_dir(normpath(join(self.batch_folder, self.train_batch_folder_name)))
        self.val_batch_folder_path = get_and_create_dir(normpath(join(self.batch_folder, self.val_batch_folder_name)))
        self.test_batch_folder_path = get_and_create_dir(normpath(join(self.batch_folder, self.test_batch_folder_name)))
        
        self.index = 0
        
        self.batchs_path_list = {self.cfg.get('Base_Header_Values','Train') : [],
                                 self.cfg.get('Base_Header_Values','Validation') : [],
                                 self.cfg.get('Base_Header_Values','Test') : []}
        
        self.list_batchs_folder = {self.cfg.get('Base_Header_Values','Train') : self.train_batch_folder_path,
                                   self.cfg.get('Base_Header_Values','Validation') : self.val_batch_folder_path,
                                   self.cfg.get('Base_Header_Values','Test') : self.test_batch_folder_path}
        
        self.initialize = False
    
    def __len__(self, base : str):
        if not self.initialize:
            raise Exception("Dataset has not been initialized")
        if self.batchs_path_list[base] == []:
            raise Exception(f"Dataset : {base} empty")
        return len(self.batchs_path_list[base])
    
    def __call__(self, base : str):
        if not self.initialize:
            raise Exception("Dataset has not been initialized")
        if self.batchs_path_list[base] == []:
            raise Exception(f"Dataset : {base} empty")
        for lr_path, label_path in self.batchs_path_list[base]:
            lr = np.load(lr_path)
            label = np.load(label_path)
            yield lr, label
    
    def __iter__(self, base : str):
        return self(base)
    
    def load_dataset(self):
        buffer = []
        for base in self.batchs_path_list:
            for file in sorted(os.listdir(self.list_batchs_folder[base])):
                filepath = join(self.list_batchs_folder[base], file)
                buffer.append(filepath)
                if len(buffer) == 2:
                    if buffer[1].split('_')[0] != buffer[0].split('_')[0]:
                        raise Exception(f"{buffer[1]} does not have the same index with {buffer[0]}")
                    if buffer[0].endswith(LR_BATCH):
                        lr_path = buffer[0]
                        label_path = buffer[1]
                    else:
                        lr_path = buffer[1]
                        label_path = buffer[0]
                    self.batchs_path_list[base].append((lr_path, label_path))
                    buffer = []
        self.initialize = True
        
    def make_and_save_dataset_batchs(self, 
                                    mri_folder, 
                                    csv_listfile_path,  
                                    batchsize,
                                    lr_downscale_factor,
                                    patchsize,
                                    step,
                                    percent_valmax,
                                    merge_hr_seg = False,
                                    save_lr = False, *args, **kwargs):
        if type(patchsize) == tuple:
            patchsize = patchsize
        else:
            patchsize = (patchsize, patchsize, patchsize)
        
        train_fp_list, val_fp_list, test_fp_list = get_hr_seg_filepath_list(mri_folder, csv_listfile_path, self.cfg)
        
        self._save_data_base_batchs(batchsize,
                                    lr_downscale_factor,
                                    percent_valmax,
                                    patchsize,
                                    step,
                                    train_fp_list, 
                                    self.train_batch_folder_path, 
                                    base = self.cfg.get('Base_Header_Values','Train'), 
                                    merge_hr_seg=merge_hr_seg,
                                    save_lr=save_lr)
        self._save_data_base_batchs(batchsize,
                                    lr_downscale_factor,
                                    percent_valmax,
                                    patchsize,
                                    step,
                                    val_fp_list, 
                                    self.val_batch_folder_path, 
                                    base = self.cfg.get('Base_Header_Values','Validation'),
                                    merge_hr_seg=merge_hr_seg,
                                    save_lr=save_lr)
        self._save_data_base_batchs(batchsize,
                                    lr_downscale_factor,
                                    percent_valmax,
                                    patchsize,
                                    step,
                                    test_fp_list, 
                                    self.test_batch_folder_path, 
                                    base = self.cfg.get('Base_Header_Values','Test'), 
                                    merge_hr_seg=merge_hr_seg,
                                    save_lr=save_lr)
        
        self.initialize = True
     
    def _save_data_base_batchs(self,
                               batchsize,
                               lr_downscale_factor,
                               percent_valmax,
                               patchsize,
                               step,
                               data_filespath_list : list, 
                               data_base_folder : str, 
                               base : str,
                               merge_hr_seg = False,
                               save_lr = False, 
                               *args, **kwargs):
        batch_index = 0
        remaining_patch = 0
        lr_gen_input_list = []
        label_dis_input_list = []
        
        for data_hr, data_seg in data_filespath_list:

            lr_img, hr_img, scaling_factor = lr_from_hr(data_hr, lr_downscale_factor, percent_valmax)
            print(f"LR mri : {lr_img}")
            print(f"HR mri : {hr_img}")
            if save_lr:
                lr_img.save_mri(join(self.batch_folder, "LR_"+basename(hr_img.filepath)))
            seg_img = read_seg(data_seg, scaling_factor)
            
            lr_gen_input, label_dis_input = create_patches_from_mri(lr_img, hr_img, seg_img, patchsize, step, merge_hr_seg = merge_hr_seg)
            print("lr patches shape : ", lr_gen_input.shape, " label patches shape : ", label_dis_input.shape)
            
            # shuffle lr_gen and hr_seg_dis here
            
            lr_gen_input_list.append(lr_gen_input)
            label_dis_input_list.append(label_dis_input)
            
            buffer_lr = np.concatenate(np.asarray(lr_gen_input_list))
            buffer_hr = np.concatenate(np.asarray(label_dis_input_list))
            buffer_lr = reshape_buffer(buffer_lr)
            buffer_hr = reshape_buffer(buffer_hr)
            
            while buffer_lr.shape[0] >= batchsize:
                
                lr_batch_name = f"{batch_index:04d}{LR_BATCH}"
                label_batch_name = f"{batch_index:04d}{LABEL_BATCH}"
                lr_batch_path = normpath(join(data_base_folder, lr_batch_name))
                label_batch_path = normpath(join(data_base_folder, label_batch_name))
                
                np.save(lr_batch_path, buffer_lr[:batchsize])
                np.save(label_batch_path, buffer_hr[:batchsize])
                
                buffer_lr = buffer_lr[batchsize:]
                buffer_hr = buffer_hr[batchsize:]

                lr_gen_input_list = [buffer_lr]
                label_dis_input_list = [buffer_hr]
                
                batch_index += 1
                
                remaining_patch = buffer_lr.shape[0]
                
                self.batchs_path_list[base].append((lr_batch_path, label_batch_path))
        
        if remaining_patch > 0:
            
            buffer_lr = np.concatenate(np.asarray(lr_gen_input_list))
            buffer_hr = np.concatenate(np.asarray(label_dis_input_list))

            buffer_lr = reshape_buffer(buffer_lr)
            buffer_hr = reshape_buffer(buffer_hr)
                
            lr_batch_name = f"{batch_index:04d}{LR_BATCH}"
            label_batch_name = f"{batch_index:04d}{LABEL_BATCH}"
            lr_batch_path = normpath(join(data_base_folder, lr_batch_name))
            label_batch_path = normpath(join(data_base_folder, label_batch_name))
                
            np.save(lr_batch_path, buffer_lr[:batchsize])
            np.save(label_batch_path, buffer_hr[:batchsize])
            
            buffer_lr = buffer_lr[remaining_patch:]
            buffer_hr = buffer_hr[remaining_patch:]
            
            lr_gen_input_list = [buffer_lr]
            label_dis_input_list = [buffer_hr]
            
            self.batchs_path_list[base].append((lr_batch_path, label_batch_path))
        
        # shuffle datas here
        
        return remaining_patch


