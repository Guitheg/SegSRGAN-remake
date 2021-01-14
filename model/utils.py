# source code from https://github.com/rousseau/deepBrain/

from tensorflow.python.ops import array_ops
from keras.engine.topology import Layer
from keras.optimizers import Optimizer
from keras.legacy import interfaces
import keras.backend as K
import numpy as np

def charbonnier_loss(y_true, y_pred):
    """
    https://en.wikipedia.org/wiki/Huber_loss
    """
    epsilon = 1e-3
    diff = y_true - y_pred
    loss = K.mean(K.sqrt(K.square(diff)+epsilon*epsilon), axis=-1)
    return K.mean(loss)

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return K.mean(y_true * y_pred)

class Activation_SegSRGAN(Layer):
    def __init__(self, int_channel=0 , seg_channel=1, activation='sigmoid', **kwargs):
        self.seg_channel = seg_channel
        self.int_channel = int_channel
        self.activation = activation
        super(Activation_SegSRGAN, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(Activation_SegSRGAN, self).build(input_shapes)

    def call(self, inputs):
        recent_input = inputs[0]
        first_input = inputs[1]
        
        if self.activation == 'sigmoid':
            segmentation = K.sigmoid(recent_input[:, self.seg_channel, :, :, :])
        else:
            raise Exception(f'do not support : {self.activation}')
        intensity = recent_input[:, self.int_channel, :, :, :]
        
        # Adding channel
        segmentation = K.expand_dims(segmentation, axis=1)
        intensity = K.expand_dims(intensity, axis=1)
        
        # residual
        residual_intensity = first_input - intensity
        return  K.concatenate([residual_intensity,segmentation], axis=1)
    
    def compute_output_shape(self, input_shapes):
        return input_shapes[0]
    
class InstanceNormalization3D(Layer):
    ''' Thanks for github.com/jayanthkoushik/neural-style 
    and https://github.com/PiscesDream/CycleGAN-keras/blob/master/CycleGAN/layers/normalization.py'''
    def __init__(self, **kwargs):
        super(InstanceNormalization3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[1],), initializer="one", trainable=True)
        self.shift = self.add_weight(name='shift', shape=(input_shape[1],), initializer="zero", trainable=True)
        super(InstanceNormalization3D, self).build(input_shape)

    def call(self, x):
        def image_expand(tensor):
            return K.expand_dims(K.expand_dims(K.expand_dims(tensor, -1), -1), -1)

        def batch_image_expand(tensor):
            return image_expand(K.expand_dims(tensor, 0))

        hwk = K.cast(x.shape[2] * x.shape[3] * x.shape[4], K.floatx())
        mu = K.sum(x, [-1, -2, -3]) / hwk
        mu_vec = image_expand(mu) 
        sig2 = K.sum(K.square(x - mu_vec), [-1, -2, -3]) / hwk
        y = (x - mu_vec) / (K.sqrt(image_expand(sig2)) + K.epsilon())

        scale = batch_image_expand(self.scale)
        shift = batch_image_expand(self.shift)
        return scale*y + shift 

    def compute_output_shape(self, input_shape):
        return input_shape

class ReflectPadding3D(Layer):
    def __init__(self, padding=1, **kwargs):
        super(ReflectPadding3D, self).__init__(**kwargs)
        self.padding = ((padding, padding), (padding, padding), (padding, padding))

    def compute_output_shape(self, input_shape):
        if input_shape[2] is not None:
            dim1 = input_shape[2] + self.padding[0][0] + self.padding[0][1]
        else:
            dim1 = None
        if input_shape[3] is not None:
            dim2 = input_shape[3] + self.padding[1][0] + self.padding[1][1]
        else:
            dim2 = None
        if input_shape[4] is not None:
            dim3 = input_shape[4] + self.padding[2][0] + self.padding[2][1]
        else:
            dim3 = None
        return (input_shape[0],
                input_shape[1],
                dim1,
                dim2,
                dim3)

    def call(self, inputs):
        pattern = [[0, 0], [0, 0], 
                   [self.padding[0][0], self.padding[0][1]],
                   [self.padding[1][0], self.padding[1][1]], 
                   [self.padding[2][0], self.padding[2][1]]]
            
        return array_ops.pad(inputs, pattern, mode= "REFLECT")

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class LR_Adam(Optimizer):
    """
    https://ksaluja15.github.io/Learning-Rate-Multipliers-in-Keras/
    
    Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., multipliers=None, **kwargs):
        super(LR_Adam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        self.epsilon = epsilon
        self.initial_decay = decay
        self.lr_multipliers = multipliers              # Multiplier for weights [0,2,4,6,...] and bias [1,3,5,7,...]
        
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs
        
        # Multiplier for weights [0,2,4,6,...] and bias [1,3,5,7,...]
        if len(params) != len(self.lr_multipliers) : 
            raise Exception("Check Multipliers !") 
        count_multipliers = 0
        
        for p, g, m, v in zip(params, grads, ms, vs):

            # Multiplier for weights [0,2,4,6,...] and bias [1,3,5,7,...]
            if self.lr_multipliers is None:
                new_lr = lr_t     
            else:
                new_lr = lr_t * self.lr_multipliers[count_multipliers]
                count_multipliers += 1
                           
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - new_lr * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(LR_Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)