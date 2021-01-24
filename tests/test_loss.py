
from model.utils import charbonnier_loss
import numpy as np
from tensorflow.keras.losses import mae

def runtest(config, *args, **kwargs):
    array_a = np.ones((1,1,2,3,4))
    array_b = np.ones((1,1,2,3,4))+50

    loss = mae(array_a, array_b)
    print(loss.shape, loss)