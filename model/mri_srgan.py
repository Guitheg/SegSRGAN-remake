
from model.utils import ReflectPadding3D, charbonnier_loss, ProgressBar
from os.path import join, normpath, isdir
from utils.files import get_and_create_dir
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LeakyReLU, Reshape, Conv3D, Add, UpSampling3D, Activation, ZeroPadding3D

def resnet_blocks(input_res, kernel, name):
    in_res_1 = ReflectPadding3D(padding=1)(input_res)
    out_res_1 = Conv3D(kernel, 3, strides=1, name=name+'_conv_a', data_format='channels_first')(in_res_1)
    out_res_1 = Activation('relu')(out_res_1)
    
    out_res_2 = ReflectPadding3D(padding=1)(out_res_1)
    out_res_2 = Conv3D(kernel, 3, strides=1, name=name+'_conv_b', data_format='channels_first')(out_res_2)
    out_res = Add()([out_res_2, input_res])
    return out_res

def make_generator_model(name : str, shape : tuple, kernel : int):
    inputs = Input(shape=(1, shape[0], shape[1], shape[2]))
    size = shape[0]
    # Representation
    gennet = ReflectPadding3D(padding=3)(inputs)
    gennet = Conv3D(kernel, 7, strides=1, name=name+'_gen_conv1', data_format='channels_first')(inputs)
    gennet = Activation('relu')(gennet)

    # # Downsampling 1
    gennet = ReflectPadding3D(padding=1)(inputs)
    gennet = Conv3D(kernel*2, 3, strides=2, name=name+'_gen_conv2', data_format='channels_first')(gennet)
    gennet = Activation('relu')(gennet)
    
    # # Downsampling 2
    gennet = ReflectPadding3D(padding=1)(inputs)
    gennet = Conv3D(kernel*4, 3, strides=2, name=name+'_gen_conv3', data_format='channels_first')(gennet)
    gennet = Activation('relu')(gennet)
            
    # # Resnet blocks : 6, 8*4 = 32
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block1')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block2')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block3')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block4')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block5')
    gennet = resnet_blocks(gennet, kernel*4, name=name+'_gen_block6')
    
    # Upsampling 1
    gennet = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(gennet)
    gennet = ReflectPadding3D(padding=1)(gennet)
    gennet = Conv3D(kernel*2, 3, strides=1, name=name+'_gen_deconv1', data_format='channels_first')(gennet)
    gennet = Activation('relu')(gennet)
    
    # Upsampling 2
    gennet = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(gennet)
    gennet = ReflectPadding3D(padding=2)(gennet)
    gennet = Conv3D(kernel, 3, strides=1, name=name+'_gen_deconv2', data_format='channels_first')(gennet)
    gennet = Activation('relu')(gennet)
    
    # Reconstruction
    gennet = ReflectPadding3D(padding=3)(gennet)
    gennet = Conv3D(1, 9, strides=1, use_bias=False, name=name+'_gen_1conv', data_format='channels_first')(gennet)
    gennet = Activation('relu')(gennet)
    
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

        self.generator = make_generator_model("generator", (16, 16, 16), 4)
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
    
    def train_step_generator(self, batch_lr, batch_hr_seg):
        
        batch_hr = batch_hr_seg[:, 0, :, :, :]
        with tf.GradientTape(persistent=True) as tape:
            batch_sr = self.generator(batch_lr, training=True)
            
            losses = {}
            losses['charbonnier'] = charbonnier_loss(batch_hr, batch_sr)
            
            total_loss = tf.add_n([l for l in losses.values()])
            
        gradients = tape.gradient(total_loss, self.generator.trainable_variables)
        self.optimizer_gen.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return losses, total_loss

    def train_step(self, batch_lr, batch_hr_seg):
        return self.train_step_generator(batch_lr, batch_hr_seg)
        
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
                
                
        