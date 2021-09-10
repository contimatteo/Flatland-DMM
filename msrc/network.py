import keras
import tensorflow as tf
from keras import Input
from keras.layers import Dense, Reshape, Layer


class PreprocessingLayer(Layer):
    def __init__(self):
        super(PreprocessingLayer, self).__init__()
        self.layers = [
            Dense(8, activation=tf.tanh),
            Dense(4, activation=tf.tanh),
            Dense(2, activation=tf.tanh),
        ]

    def call(self, input_tensor, training=False, **kwargs):
        x = input_tensor
        for layer in self.layers:
            x = layer(x, training=training)
        return x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class MainNet(Layer):
    def __init__(self):
        super(MainNet, self).__init__()
        self.layers = [
            Dense(15, activation=tf.tanh),
            Dense(9, activation=tf.tanh),
            Dense(3, activation=tf.tanh),
        ]

    def call(self, input_tensor, training=False, **kwargs):
        x = input_tensor
        for layer in self.layers:
            x = layer(x, training=training)
        return x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class Argmax(Layer):
    """
    Based on https://github.com/YerevaNN/R-NET-in-Keras/blob/master/layers/Argmax.py
    """

    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        return tf.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_model(input_shape):
    inputs = Input(shape=input_shape)
    pre_processing = PreprocessingLayer()
    reshaper = Reshape((input_shape[0], input_shape[1] * 2))
    main_net = MainNet()
    action_chooser = Argmax()

    x = pre_processing(inputs)
    x = reshaper(x)
    x = main_net(x)
    x = action_chooser(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model
