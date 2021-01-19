import SimpleITK as sitk
import numpy as np
from os.path import normpath, abspath, isfile, dirname, isdir


class MRI(object):
    EXTENSIONS = [".nii.gz", ".nrrd", ".nii"]
    def __init__(self, *args, **kwwargs):
        self.filepath = None
        self.sitk_instance = None
        self.img_array = None
        self.resolution = None
        self.origin = None
        self.direction = None
        
    def load_from_file(self, filepath : str):
        
        filepath = abspath(normpath(filepath))
        if not isfile(filepath):
            raise Exception(f'Le fichier {filepath} est introuvable')
        
        self._check_file_extension(filepath)
        
        self.filepath = filepath
                    
        self.sitk_instance = sitk.ReadImage(self.filepath)
        self.resolution = self.sitk_instance.GetSpacing()
        self.origin = self.sitk_instance.GetOrigin()
        self.direction = self.sitk_instance.GetDirection()
        
        self._set_parameter()
        
    def load_from_array(self, img_array : np.ndarray, resolution : tuple, origin : list, direction : list):
        self.sitk_instance = sitk.GetImageFromArray(img_array)
        self.sitk_instance.SetSpacing(resolution)
        self.sitk_instance.SetOrigin(origin)
        self.sitk_instance.SetDirection(direction)
        self._set_parameter()   
            
    def save_mri(self, path_to_save : str):
        self._check_instance_load()
        
        filepath = abspath(normpath(path_to_save))
        
        dir = dirname(filepath)
        if not isdir(dir):
            raise Exception(f"Le dossier {dir} est introuvable.")
        
        self._check_file_extension(filepath)
        
        self.filepath = filepath
        
        sitk.WriteImage(self.sitk_instance, path_to_save)
    
    def get_img_array(self):
        self._check_instance_load()
        return self.img_array
    
    def get_resolution(self):
        self._check_instance_load()
        return self.resolution
    
    def get_direction(self):
        self._check_instance_load()
        return self.direction
    
    def get_origin(self):
        self._check_instance_load()
        return self.origin
    
    def set_img_array(self, img_array : np.ndarray):
        self._check_instance_load()
        self.img_array = img_array
    
    def _set_parameter(self):
        self._check_instance_load()
        self.img_array = sitk.GetArrayFromImage(self.sitk_instance)
        self.resolution = self.sitk_instance.GetSpacing()
        self.origin = self.sitk_instance.GetOrigin()
        self.direction = self.sitk_instance.GetDirection()
    
    def _check_instance_load(self):
        if self.sitk_instance is None:
            raise Exception("MRI or sITK instance has not been loaded (use load_from_file or load_from_array)")
        
    def _check_file_extension(self, filepath):
        if all([not filepath.endswith(ext) for ext in MRI.EXTENSIONS]):
            raise Exception(f'filepath has to be one of this extensions file: {MRI.EXTENSIONS} (value : {filepath})')
        
    def __call__(self, array : np.ndarray = None):
        if not array is None:
            self.set_img_array(array)
        return self.get_img_array()
    
    def __repr__(self):
        return self.__str__()    
    
    def __str__(self):
        if self.sitk_instance is None:
            return "MRI not loaded"
        else:
            return f"shape : {self.img_array.shape}, resolution : {self.resolution}, path : {self.filepath}"