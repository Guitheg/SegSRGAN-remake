from tensorflow.keras.layers import Layer
from tensorflow.python.ops import array_ops

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