
from genericpath import exists
from dataset.dataset_manager import MRI_Dataset
from model.gan_model import GAN_Model
from model.utils import LR_Adam, ReflectPadding3D, InstanceNormalization3D, Activation_SegSRGAN, gradient_penalty_loss, wasserstein_loss, charbonnier_loss
import numpy as np
from functools import partial
from keras.models import Model
from keras.layers import Input, LeakyReLU, Reshape
from keras.layers import Conv3D, Add, UpSampling3D, Activation
from keras.optimizers import Adam
from keras.initializers import lecun_normal
import tensorflow as tf

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

class SegSRGAN(GAN_Model):
    
    def __init__(self, 
                 checkpoints_folder : str,
                 shape : tuple,
                 lambda_rec : float = 1,
                 lambda_adv : float = 0.001,
                 lambda_gp : float = 10,
                 dis_kernel : int = 32,
                 gen_kernel : int = 16,
                 lr_dismodel : float = 0.0001,
                 lr_genmodel : float = 0.0001,
                 max_checkpoints_to_keep : int = 2,
                 *args, **kwargs):
        super(SegSRGAN, self).__init__(checkpoints_folder, 
                                       max_checkpoints_to_keep, 
                                       shape=shape,
                                       dis_kernel=dis_kernel, 
                                       gen_kernel=gen_kernel, 
                                       *args, **kwargs)
        
        self.generator_trainer = self.make_generator_trainer(shape, lr_genmodel, lambda_adv, lambda_rec)
        self.discriminator_trainer = self.make_discriminator_trainer(shape, lr_dismodel, lambda_gp)
    
    def make_generator_model(self, shape, gen_kernel, *args, **kwargs):
        return segsrgan_generator_block('Generator', shape, gen_kernel)
    
    def make_generator_trainer(self, shape, lr_genmodel, lambda_adv, lambda_rec, *args, **kwargs):
        input_gen = Input(shape=(1, shape[0], shape[1], shape[2]), name='input_gen')
        gen = self.generator(input_gen)
        fool = self.discriminator(gen)
        
        generator_trainer = Model(input_gen, [fool, gen])
        generator_trainer.compile(LR_Adam(lr=lr_genmodel,
                                          beta_1=0.5,
                                          beta_2=0.999),
                                  loss=[wasserstein_loss, charbonnier_loss],
                                  loss_weights=[lambda_adv, lambda_rec])
        return generator_trainer
         
    def make_discriminator_model(self, shape, dis_kernel, *args, **kwargs): 
        return segsrgan_discriminator_block('Discriminator', shape, dis_kernel)
    
    def make_discriminator_trainer(self, shape, lr_dismodel, lambda_gp):
        real_dis = Input(shape=(1, shape[0], shape[1], shape[2]), name='real_dis')
        fake_dis = Input(shape=(1, shape[0], shape[1], shape[2]), name='fake_dis')       
        interp_dis = Input(shape=(1, shape[0], shape[1], shape[2]), name='interp_dis') 
        
        real_decision = self.discriminator()(real_dis)
        fake_decision = self.discriminator()(fake_dis)
        interp_decision = self.discriminator()(interp_dis)
        
        partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interp_dis,
                          gradient_penalty_weight=lambda_gp)
        partial_gp_loss.__name__ = 'gradient_penalty'
    
        discriminator_trainer = Model([real_dis, fake_dis, interp_dis], 
                              [real_decision, fake_decision, interp_decision])
        discriminator_trainer.compile(Adam(lr=self.lr_DisModel, 
                                   beta_1=0.5, 
                                   beta_2=0.999),
                            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                            loss_weights=[1, 1, 1])
        return discriminator_trainer 
    
    def _fit_one_epoch(self, dataset, *args, **kwargs):
        n_critic = 5
        if not kwargs['n_critic'] is None:
            n_critic = kwargs['n_critic']
        
        for lr, hr_seg in dataset:
            dis_losses = self.fit_one_step_discriminator(n_critic, hr_seg, lr)
            gen_loss = self.fit_one_step_generator(hr_seg, lr)
            
    def fit_one_step_discriminator(self, n_critic, batch_real, batch_gen_inp, *args, **kwargs):
        batchsize = batch_real.shape[0]
        real = -np.ones([batchsize, 1], dtype=np.float32)
        fake = -real
        dummy = np.zeros([batchsize, 1], dtype=np.float32)
        dis_losses = []
        
        for _ in range(n_critic):
            
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
        gen_loss = self.generator_trainer.train_on_batch([batch_real],
                                                         [real, batch_gen_inp])
        
        return gen_loss
                
    
        