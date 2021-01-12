from configparser import ConfigParser
import os
from pandas import Series
import pandas as pd
from os.path import join, normpath

def os_path_join_for_pandas(add_this_path : str, to_all_lign_of_this_column : Series):
    return to_all_lign_of_this_column.apply(lambda filepath : normpath(join(add_this_path, filepath)))

def two_columns_to_tuple_list(x, cola, colb):
    _, row = x
    return row[cola], row[colb]

def get_hr_seg_filepath_list(mri_folder : str, csv_listfile_path : str, cfg : ConfigParser):
    hr_header = cfg.get('CSV_Header','HR_Header')
    seg_header = cfg.get('CSV_Header','Seg_Header')
    base_Header = cfg.get('CSV_Header', 'Base_Header')
    train_base = cfg.get('Base_Header_Values', 'Train')
    val_base = cfg.get('Base_Header_Values', 'Validation')
    test_base = cfg.get('Base_Header_Values', 'Test')
    
    maindata_filespath_df = pd.read_csv(csv_listfile_path)
    
    maindata_filespath_df[hr_header] = os_path_join_for_pandas(mri_folder, maindata_filespath_df[hr_header])
    maindata_filespath_df[seg_header] = os_path_join_for_pandas(mri_folder, maindata_filespath_df[seg_header])
    
    traindata_filespath_df = maindata_filespath_df[maindata_filespath_df[base_Header] == train_base]
    valdata_filespath_df = maindata_filespath_df[maindata_filespath_df[base_Header] == val_base]
    testdata_filespath_df = maindata_filespath_df[maindata_filespath_df[base_Header] == test_base]

    # list : [ (hr1path, seg1path), (hr2path, seg2path), ...]
    train_hr_seg_filepath_list = list(map((lambda x : two_columns_to_tuple_list(x, hr_header, seg_header)), 
                                          traindata_filespath_df.iterrows()))
    val_hr_seg_filepath_list = list(map((lambda x : two_columns_to_tuple_list(x, hr_header, seg_header)), 
                                          valdata_filespath_df.iterrows()))
    test_hr_seg_filepath_list = list(map((lambda x : two_columns_to_tuple_list(x, hr_header, seg_header)), 
                                          testdata_filespath_df.iterrows()))

    return train_hr_seg_filepath_list, val_hr_seg_filepath_list, test_hr_seg_filepath_list

def get_and_create_dir(dirpath : str):
    if os.path.isfile(dirpath):
        raise Exception(f"{dirpath} is not a folder but a file")
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    return os.path.normpath(dirpath) 