import os
from typing import Union
import tensorflow as tf

class GAN_Model(object):
    def __init__(self, 
                 checkpoints_folder : str, 
                 max_checkpoints_to_keep : int = 2, 
                 *args, **kwargs):
        
        self.discriminator = self.make_discriminator_model(*args, **kwargs)  
        self.generator = self.make_generator_model(*args, **kwargs)
        
        self.checkpoints_folder = checkpoints_folder
        self.checkpoint_epoch = 0
        self.checkpoint = tf.train.Checkpoint(  epoch = self.checkpoint_epoch,
                                                generator_optimizer=self.generator_optimizer,
                                                discriminator_optimizer=self.discriminator_optimizer,
                                                generator=self.generator,
                                                discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 
                                                             directory=self.checkpoints_folder, 
                                                             max_to_keep=max_checkpoints_to_keep)
        
    def make_discriminator_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def make_generator_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def fit(self, 
            dataset,
            n_epochs : int = 1,
            *args, **kwargs):
        
        self._load_checkpoint()
        
        for epoch in range(self.checkpoint_epoch, n_epochs):
            print(f"Epoch {epoch+1} / {n_epochs} : ")
            self._fit_one_epoch(dataset, *args, **kwargs)
            self._save_checkpoint()    
    
    def _load_checkpoint(self, *args, **kwargs):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            
    def _save_checkpoint(self, *args, **kwargs):
        self.checkpoint_manager.save()
    
    def _fit_one_epoch(self, dataset, *args, **kwargs):
        raise NotImplementedError