
from utils.files import get_and_create_dir
from utils.patches import test_by_patch
from dataset.dataset_manager import MRI_Dataset
from model.utils import LR_Adam, Activation_SegSRGAN, gradient_penalty_loss, wasserstein_loss, charbonnier_loss
from layers.reflect_padding import ReflectPadding3D
from layers.instance_normalization import InstanceNormalization3D
import numpy as np
from os.path import join, normpath, isdir, basename
from functools import partial
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LeakyReLU, Reshape, Conv3D, Add, UpSampling3D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import lecun_normal
import tensorflow as tf
from tensorflow.keras import backend as K
from os.path import join, normpath, isfile

def resnet_blocks(input_res, kernel, name):
    gen_initializer = lecun_normal()
    in_res_1 = ReflectPadding3D(padding=1)(input_res)
    out_res_1 = Conv3D(kernel, 3, strides=1, kernel_initializer=gen_initializer, 
                       use_bias=False,
                       name=name+'_conv_a', 
                       data_format='channels_first')(in_res_1)
    out_res_1 = InstanceNormalization3D(name=name+'_isnorm_a')(out_res_1)
    out_res_1 = Activation('relu')(out_res_1)
    
    in_res_2 = ReflectPadding3D(padding=1)(out_res_1)
    out_res_2 = Conv3D(kernel, 3, strides=1, kernel_initializer=gen_initializer, 
                       use_bias=False,
                       name=name+'_conv_b', 
                       data_format='channels_first')(in_res_2)
    out_res_2 = InstanceNormalization3D(name=name+'_isnorm_b')(out_res_2)
    
    out_res = Add()([out_res_2,input_res])
    return out_res

def segsrgan_generator_block(name : str, shape : tuple, kernel : int):
    gen_initializer = lecun_normal()
    inputs = Input(shape=(1, shape[0], shape[1], shape[2]))

    # Representation
    gennet = ReflectPadding3D(padding=3)(inputs)
    gennet = Conv3D(kernel, 7, strides=1, kernel_initializer=gen_initializer, 
                    use_bias=False,
                    name=name+'_gen_conv1', 
                    data_format='channels_first')(gennet)
    gennet = InstanceNormalization3D(name=name+'_gen_isnorm_conv1')(gennet)
    gennet = Activation('relu')(gennet)

    # Downsampling 1
    gennet = ReflectPadding3D(padding=1)(gennet)
    gennet = Conv3D(kernel*2, 3, strides=2, kernel_initializer=gen_initializer, 
                    use_bias=False,
                    name=name+'_gen_conv2', 
                    data_format='channels_first')(gennet)
    gennet = InstanceNormalization3D(name=name+'_gen_isnorm_conv2')(gennet)
    gennet = Activation('relu')(gennet)
    
    # Downsampling 2
    gennet = ReflectPadding3D(padding=1)(gennet)
    gennet = Conv3D(kernel*4, 3, strides=2, kernel_initializer=gen_initializer, 
                    use_bias=False,
                    name=name+'_gen_conv3', 
                    data_format='channels_first')(gennet)
    gennet = InstanceNormalization3D(name=name+'_gen_isnorm_conv3')(gennet)
    gennet = Activation('relu')(gennet)
            
    # Resnet blocks : 6, 8*4 = 32
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block1')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block2')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block3')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block4')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block5')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block6')
    
    # Upsampling 1
    gennet = UpSampling3D(size=(2, 2, 2), 
                            data_format='channels_first')(gennet)
    gennet = ReflectPadding3D(padding=1)(gennet)
    gennet = Conv3D(kernel*2, 3, strides=1, kernel_initializer=gen_initializer, 
                    use_bias=False,
                    name=name+'_gen_deconv1', 
                    data_format='channels_first')(gennet)
    gennet = InstanceNormalization3D(name=name+'_gen_isnorm_deconv1')(gennet)
    gennet = Activation('relu')(gennet)
    
    # Upsampling 2
    gennet = UpSampling3D(size=(2, 2, 2), 
                            data_format='channels_first')(gennet)
    gennet = ReflectPadding3D(padding=1)(gennet)
    gennet = Conv3D(kernel, 3, strides=1, kernel_initializer=gen_initializer,
                    use_bias=False,
                    name=name+'_gen_deconv2', 
                    data_format='channels_first')(gennet)
    gennet = InstanceNormalization3D(name=name+'_gen_isnorm_deconv2')(gennet)
    gennet = Activation('relu')(gennet)
    
    # Reconstruction
    gennet = ReflectPadding3D(padding=3)(gennet)
    gennet = Conv3D(2, 7, strides=1, kernel_initializer=gen_initializer, 
                    use_bias=False,
                    name=name+'_gen_1conv', 
                    data_format='channels_first')(gennet)
    
    predictions = gennet
    predictions = Activation_SegSRGAN()([predictions, inputs])
    
    model = Model(inputs=inputs, outputs=predictions, name=name)
    return model

def segsrgan_discriminator_block(name : str, shape : tuple, kernel : int):
    # In:
    inputs = Input(shape=(2, shape[0], shape[1], shape[2]), name='dis_input')
    
    # Input 64
    disnet = Conv3D(kernel*1, 4, strides=2, 
                    padding = 'same',
                    kernel_initializer='he_normal', 
                    data_format='channels_first', 
                    name=name+'_conv_dis_1')(inputs)
    disnet = LeakyReLU(0.01)(disnet)
    
    # Hidden 1 : 32
    disnet = Conv3D(kernel*2, 4, strides=2, 
                    padding = 'same',
                    kernel_initializer='he_normal', 
                    data_format='channels_first', 
                    name=name+'_conv_dis_2')(disnet)
    disnet = LeakyReLU(0.01)(disnet) 
    
    # Hidden 2 : 16
    disnet = Conv3D(kernel*4, 4, strides=2, 
                    padding = 'same',
                    kernel_initializer='he_normal', 
                    data_format='channels_first', 
                    name=name+'_conv_dis_3')(disnet)
    disnet = LeakyReLU(0.01)(disnet)
    
    
    # Hidden 3 : 8
    disnet = Conv3D(kernel*8, 4, strides=2, 
                    padding = 'same',
                    kernel_initializer='he_normal',
                    data_format='channels_first', 
                    name=name+'_conv_dis_4')(disnet)
    disnet = LeakyReLU(0.01)(disnet)
    
    
    # Hidden 4 : 4
    disnet = Conv3D(kernel*16, 4, strides=2, 
                    padding = 'same',
                    kernel_initializer='he_normal',
                    data_format='channels_first', 
                    name=name+'_conv_dis_5')(disnet)
    disnet = LeakyReLU(0.01)(disnet)
    
    # Decision : 2
    decision = Conv3D(1, 2, strides=1, 
                    use_bias=False,
                    kernel_initializer='he_normal',
                    data_format='channels_first', 
                    name='dis_decision')(disnet) 
    decision = Reshape((1,))(decision)
    
    model = Model(inputs=[inputs], outputs=[decision], name=name)
    
    return model

class SegSRGAN():
    
    def __init__(self,
                 name : str,
                 checkpoint_folder : str,
                 weight_folder : str,
                 logs_folder : str,
                 shape : tuple = (64, 64, 64),
                 lambda_rec : float = 1,
                 lambda_adv : float = 0.001,
                 lambda_gp : float = 10,
                 dis_kernel : int = 32,
                 gen_kernel : int = 16,
                 lr_dismodel : float = 0.0001,
                 lr_genmodel : float = 0.0001,
                 max_checkpoints_to_keep : int = 2,
                 *args, **kwargs):

        self.name = name
        self.generator = self.make_generator_model(shape, gen_kernel, *args, **kwargs)
        self.generator.summary()
        
        self.discriminator = self.make_discriminator_model(shape, dis_kernel, *args, **kwargs)  
        self.discriminator.summary()
        
        self.generator_trainer = self.make_generator_trainer(shape, lr_genmodel, lambda_adv, lambda_rec)
        self.discriminator_trainer = self.make_discriminator_trainer(shape, lr_dismodel, lambda_gp)
        
        if not isdir(checkpoint_folder):
            raise Exception(f"Checkpoint's folder unknow : {checkpoint_folder}")
        else:  
            self.checkpoints_folder = get_and_create_dir(normpath(join(checkpoint_folder, name)))
        
        if not isdir(weight_folder):
            raise Exception(f"Weight's folder unknow : {weight_folder}")
        else:  
            self.weight_folder = get_and_create_dir(normpath(join(weight_folder, name)))
            
        if not isdir(logs_folder):
            raise Exception(f" Logs's folder unknow : {logs_folder}")
        else:  
            self.logs_folder = get_and_create_dir(normpath(join(logs_folder, name)))      
        
        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0, name='epoch'),
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=self.checkpoints_folder,
                                                             max_to_keep=3)
        
        
    def predict(self, patches):
        sr_seg = self.generator.predict(patches)
        return sr_seg

    def train(self, 
            dataset : MRI_Dataset,
            n_epochs : int = 1,
            mri_to_visualize=None, 
            output_dir=None,
            *args, **kwargs):
        if output_dir:
            output_dir = get_and_create_dir(join(output_dir, self.name))
            
        status = self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        
        for epoch in range(0, n_epochs):
            print(f"Epoch {epoch+1} / {n_epochs} : ")
            self._fit_one_epoch(dataset('Train'), *args, **kwargs)
            self.checkpoint_manager.save()
        
        if mri_to_visualize:
                if output_dir is None:
                    raise Exception("You should specify the directory of output")
                sr_mri = test_by_patch(mri_to_visualize, self)
                sr_mri.save_mri(join(output_dir, self.name+"_epoch_"+str(1)+"_SR_"+basename(mri_to_visualize.filepath)))
    
    def make_generator_model(self, shape, gen_kernel, *args, **kwargs):
        return segsrgan_generator_block('Generator', shape, gen_kernel)
    
    def make_generator_trainer(self, shape, lr_genmodel, lambda_adv, lambda_rec, *args, **kwargs):
        input_gen = Input(shape=(1, shape[0], shape[1], shape[2]), name='input_gen')
        gen = self.generator(input_gen)
        fool = self.discriminator(gen)
        
        # freezing discriminator weights
        all_parameters = 63
        generator_parameters = 52
        multipliers = np.ones(all_parameters)
        for idx in range(generator_parameters, all_parameters):
            multipliers[idx]= 0.0
        
        generator_trainer = Model(input_gen, [fool, gen])
        generator_trainer.compile(LR_Adam(lr=lr_genmodel,
                                          beta_1=0.5,
                                          beta_2=0.999,
                                          multipliers=multipliers),
                                  loss=[wasserstein_loss, charbonnier_loss],
                                  loss_weights=[lambda_adv, lambda_rec])
        return generator_trainer
         
    def make_discriminator_model(self, shape, dis_kernel, *args, **kwargs): 
        return segsrgan_discriminator_block('Discriminator', shape, dis_kernel)
    
    def make_discriminator_trainer(self, shape, lr_dismodel, lambda_gp):
        real_dis = Input(shape=(2, shape[0], shape[1], shape[2]), name='real_dis')
        fake_dis = Input(shape=(2, shape[0], shape[1], shape[2]), name='fake_dis')       
        interp_dis = Input(shape=(2, shape[0], shape[1], shape[2]), name='interp_dis') 
        
        real_decision = self.discriminator(real_dis)
        fake_decision = self.discriminator(fake_dis)
        interp_decision = self.discriminator(interp_dis)
        
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interp_dis,
                          gradient_penalty_weight=lambda_gp)
        partial_gp_loss.__name__ = 'gradient_penalty'
    
        discriminator_trainer = Model([real_dis, fake_dis, interp_dis], 
                              [real_decision, fake_decision, interp_decision])
        discriminator_trainer.compile(Adam(lr=lr_dismodel, 
                                   beta_1=0.5, 
                                   beta_2=0.999),
                            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                            loss_weights=[1, 1, 1])
        return discriminator_trainer 
    
    def _fit_one_epoch(self, dataset_train, *args, **kwargs):
        n_critic = 5
        if 'n_critic' in kwargs:
            n_critic = kwargs['n_critic']
        
        for lr, hr_seg in dataset_train:
            dis_losses = self.fit_one_step_discriminator(n_critic, hr_seg, lr)
            gen_loss = self.fit_one_step_generator(hr_seg, lr)
            
    def fit_one_step_discriminator(self, n_critic, batch_real, batch_gen_inp, *args, **kwargs):
        batchsize = batch_real.shape[0]
        real = -np.ones([batchsize, 1], dtype=np.float32)
        fake = -real
        dummy = np.zeros([batchsize, 1], dtype=np.float32)
        dis_losses = []
        
        for idx_dis_step in range(n_critic):
            print(f"{idx_dis_step} / {n_critic}")
            # Fake image from generator and Interpolated image generation : 
            epsilon = np.random.uniform(0, 1, size=(batchsize, 2, 1, 1, 1))
            batch_generated = self.generator.predict(batch_gen_inp)
            batch_interpolated = epsilon*batch_real + (1-epsilon)*batch_generated

            # Train discriminator
            dis_loss = self.discriminator_trainer.train_on_batch([batch_real, batch_generated, batch_interpolated],
                                                                 [real, fake, dummy])
            dis_losses.append(dis_loss)
        
        return dis_losses
            
    def fit_one_step_generator(self, batch_real, batch_gen_inp, *args, **kwargs):
        batchsize = batch_real.shape[0]
        real = -np.ones([batchsize, 1], dtype=np.float32)
        
        # Train generator
        gen_loss = self.generator_trainer.train_on_batch([batch_gen_inp],
                                                         [real, batch_real])
        
        return gen_loss
                
    
        