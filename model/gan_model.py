import os
from typing import Union
import tensorflow as tf

class GAN_Model(object):
    def __init__(self, 
                 checkpoints_folder : str, 
                 max_checkpoints_to_keep : int = 2, 
                 *args, **kwargs):
        
        self.discriminator = self.make_discriminator_model(*args, **kwargs)
        self.discriminator_loss = self.make_discriminator_loss(*args, **kwargs)
        self.discriminator_optimizer = self.make_discriminator_optimizer(*args, **kwargs)
        
        self.generator = self.make_generator_model(*args, **kwargs)
        self.generator_loss = self.make_generator_loss(*args, **kwargs)
        self.generator_optimizer = self.make_generator_optimizer(*args, **kwargs)
        
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
    
    def make_discriminator_loss(self, *args, **kwargs):
        raise NotImplementedError
    
    def make_discriminator_optimizer(self, *args, **kwargs):
        raise NotImplementedError
    
    def make_generator_model(self, *args, **kwargs):
        raise NotImplementedError
    
    def make_generator_loss(self, *args, **kwargs):
        raise NotImplementedError
    
    def make_generator_optimizer(self, *args, **kwargs):
        raise NotImplementedError
    
    def fit(self, 
            dataset,
            n_epochs : int = 1, 
            batchsize : int = 1,
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
            
    @tf.function                      
    def _fit_one_step(self, batch_real_inp, batch_gen_inp, *args, **kwargs):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            batch_generated = self.generator(batch_gen_inp, training=True)
            
            batch_real_output = self.discriminator(batch_real_inp, training=True)
            batch_fake_output = self.discriminator(batch_generated, training=True)
            
            batch_gen_loss = self.generator_loss(batch_fake_output)
            batch_dis_loss = self.discriminator_loss(batch_real_output, batch_fake_output)
            
        batch_gen_gradients = gen_tape(batch_gen_loss, self.generator.trainable_variables)
        batch_dis_gradients = dis_tape(batch_dis_loss, self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(batch_gen_gradients, self.generator.trainable_variable))
        self.discriminator_optimizer.apply_gradients(zip(batch_dis_gradients, self.discriminator.trainable_variables))
                
    
        