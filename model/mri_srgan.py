
from model.utils import charbonnier_loss, ProgressBar
from layers.reflect_padding import ReflectPadding3D
from layers.instance_normalization import InstanceNormalization3D
from os.path import join, normpath, isdir
from utils.files import get_and_create_dir
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LeakyReLU, Reshape, Conv3D, Add, UpSampling3D, Activation, ZeroPadding3D

def resnet_blocks(input_res, kernel, name, initializer):
    in_res_1 = ReflectPadding3D(padding=1, name=name+"_reflect_padding_a")(input_res)
    out_res_1 = Conv3D(kernel, 3, strides=1, use_bias=False, name=name+'_conv_a', kernel_initializer=initializer, data_format='channels_first')(in_res_1)
    out_res_1 = InstanceNormalization3D(name=name + '_norm_a')(out_res_1)
    out_res_1 = Activation('relu', name=name + '_relu_a')(out_res_1)
    
    out_res_2 = ReflectPadding3D(padding=1, name=name+"_reflect_padding_b")(out_res_1)
    out_res_2 = Conv3D(kernel, 3, strides=1, use_bias=False, name=name+'_conv_b', kernel_initializer=initializer, data_format='channels_first')(out_res_2)
    out_res_2 = InstanceNormalization3D(name=name + '_norm_b')(out_res_2)
    out_res = Add()([out_res_2, input_res])
    return out_res

def downsampling_block(inputs, n_channels, kernel_size, strides, padding, name, kernel_initializer):
    output = ReflectPadding3D(padding=padding, name=name+"_reflect_padding")(inputs)
    output = Conv3D(n_channels, kernel_size, strides=strides, use_bias=False, name=name+'_conv', kernel_initializer=kernel_initializer, data_format='channels_first')(output)
    output = InstanceNormalization3D(name=name+'_norm')(output)
    output = Activation('relu', name=name+"_relu")(output)
    return output

def upsampling_block(inputs, n_channels, kernel_size, strides, padding, name, kernel_initializer, upscale_factor):
    output = UpSampling3D(size=upscale_factor, data_format='channels_first', name=name+"_upscale")(inputs)
    output = ReflectPadding3D(padding=padding, name=name+"_reflect_padding")(output)
    output = Conv3D(n_channels, kernel_size, strides=strides, use_bias=False, name=name+'_conv', kernel_initializer=kernel_initializer, data_format='channels_first')(output)
    output = InstanceNormalization3D(name=name + '_norm')(output)
    output = Activation('relu', name=name+"_relu")(output)
    return output

def make_generator_model(name : str, shape : tuple, kernel : int):
    lecun_init = tf.keras.initializers.LecunNormal()

    inputs = Input(shape=(1, shape[0], shape[1], shape[2]))

    # Representation
    gennet = downsampling_block(inputs, n_channels=kernel, kernel_size=7, strides=1, padding=3, name=name+"_downsample1", kernel_initializer=lecun_init)

    # # Downsampling 1
    gennet = downsampling_block(gennet, n_channels=kernel*2, kernel_size=3, strides=2, padding=1, name=name+"_downsample2", kernel_initializer=lecun_init)
    
    # # Downsampling 2
    gennet = downsampling_block(gennet, n_channels=kernel*4, kernel_size=3, strides=2, padding=1, name=name+"_downsample3", kernel_initializer=lecun_init)
            
    # # Resnet blocks : 6, 8*4 = 32
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_resblock1', initializer=lecun_init)
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_resblock2', initializer=lecun_init)
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_resblock3', initializer=lecun_init)
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_resblock4', initializer=lecun_init)
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_resblock5', initializer=lecun_init)
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_resblock6', initializer=lecun_init)
    
    # Upsampling 1
    gennet = upsampling_block(gennet, n_channels=kernel*2, kernel_size=3, strides=1, padding=1, name=name+"_upsample1", kernel_initializer=lecun_init, upscale_factor=(2, 2, 2))
    
    # Upsampling 2
    gennet = upsampling_block(gennet, n_channels=kernel, kernel_size=3, strides=1, padding=1, name=name+"_upsample2", kernel_initializer=lecun_init, upscale_factor=(2, 2, 2))
    
    # Reconstruction
    gennet = ReflectPadding3D(padding=3, name=name+'_reconstruction_reflect_padding')(gennet)
    gennet = Conv3D(1, 7, strides=1, use_bias=False, name=name+'_reconstruction_conv', kernel_initializer=lecun_init, data_format='channels_first')(gennet)
    gennet = Activation('sigmoid', name=name+'_reconstruction_sigmoid')(gennet)
    
    model = Model(inputs=inputs, outputs=gennet, name=name)
    return model
    

class MRI_SRGAN():
    
    def __init__(self, name : str, 
                 checkpoint_folder : str,
                 logs_folder : str,
                 make_generator_model=make_generator_model,
                 make_discriminator_model=None,
                 *args, **kwargs):
        
        self.name = name
        
        if K.backend() == "tensorflow":
            from tensorflow.python.client import device_lib
            print(device_lib.list_local_devices())
        
        if not isdir(checkpoint_folder):
            raise Exception(f"Checkpoint's folder unknow : {checkpoint_folder}")
        else:  
            self.checkpoint_folder = get_and_create_dir(normpath(join(checkpoint_folder, name)))
            
        if not isdir(logs_folder):
            raise Exception(f"logs's folder unknow : {logs_folder}")
        else:  
            self.logs_folder = get_and_create_dir(normpath(join(logs_folder, name)))    
        
        self.optimizer_gen = keras.optimizers.Adam()

        self.generator = make_generator_model("gen", (32, 32, 32), 4)
        self.generator.summary()
        
        
        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0, name='step'),
                                              optimizer_G=self.optimizer_gen,
                                              model=self.generator)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=self.checkpoint_folder,
                                                             max_to_keep=3)
        
        self.load_checkpoint()
        
        self.summary_writer = tf.summary.create_file_writer(self.logs_folder)
        
    def load_checkpoint(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Load ckpt from {} at epoch {}.'.format(
                self.checkpoint_manager.latest_checkpoint, 
                self.checkpoint.epoch.numpy()))
        else:
            print("Training from scratch.")
    
    def train_step_generator(self, batch_lr, batch_hr):

        with tf.GradientTape(persistent=True) as tape:
            batch_sr = self.generator(batch_lr, training=True)
            
            losses = {}
            losses['charbonnier'] = charbonnier_loss(batch_hr, batch_sr)
            
            total_loss = tf.add_n([l for l in losses.values()])
            
        gradients = tape.gradient(total_loss, self.generator.trainable_variables)
        self.optimizer_gen.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return losses, total_loss

    def train_step(self, batch_lr, batch_hr):
        return self.train_step_generator(batch_lr, batch_hr)
        
    def train(self, dataset, n_epochs):
        prog_bar = ProgressBar(n_epochs, self.checkpoint.epoch.numpy())
        remaining_epochs = n_epochs - self.checkpoint.epoch.numpy()
        for _ in range(remaining_epochs):
            print(f"Epoch : {self.checkpoint.epoch.numpy()}/{n_epochs}")
            for step, (lr, hr_seg) in enumerate(dataset('Train')):
                # first channel : hr
                losses, total_loss = self.train_step_generator(lr, hr_seg)
                prog_bar.update("loss={:.4f}".format(total_loss.numpy()))
                
                if step % (int(dataset.__len__('Train')//100)) == 0:    
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss_G/total_loss', total_loss, step=step)

                        for k, l in losses.items():
                            tf.summary.scalar('loss_G/{}'.format(k), l, step=step)
                            
                        tf.summary.scalar('learning_rate_G', self.optimizer_gen.lr(step), step=step)
                        
            self.checkpoint_manager.save()
            print("\nSave ckpt file at {}".format(self.checkpoint_manager.latest_checkpoint))     
            self.checkpoint.epoch.assign_add(1)
            
        print("Training done !") 
                
                
        