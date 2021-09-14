import keras
import tensorflow as tf
from keras import Input
from keras.layers import Dense, Reshape, Layer


class CustomSequentialLayer(Layer):
    def __init__(self, layers):
        super(CustomSequentialLayer, self).__init__()
        self.layers = layers

    def call(self, input_tensor, training=False, **kwargs):
        x = input_tensor
        for layer in self.layers:
            x = layer(x, training=training)
        return x

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"layers": self.layers})
        return config


class PreprocessingLayer(CustomSequentialLayer):
    def __init__(self):
        self.out_dim = 4
        super(PreprocessingLayer, self).__init__([
            Dense(8, activation=tf.tanh),
            Dense(6, activation=tf.tanh),
            Dense(4, activation=tf.tanh),
            Dense(self.out_dim, activation=tf.tanh),
        ])


class MainNet(CustomSequentialLayer):
    def __init__(self):
        self.out_dim = 1
        super(MainNet, self).__init__([
            Dense(15, activation=tf.tanh),
            Dense(15, activation=tf.tanh),
            Dense(9, activation=tf.tanh),
            Dense(3, activation=tf.tanh),
            Dense(self.out_dim, activation=tf.sigmoid),
        ])


def get_model(input_shape):
    print(input_shape)
    inputs = Input(shape=input_shape)
    pre_processing = PreprocessingLayer()
    reshape1 = Reshape((input_shape[0], input_shape[1] * pre_processing.out_dim))
    main_net = MainNet()
    reshape2 = Reshape((input_shape[0] * main_net.out_dim, ))

    x = pre_processing(inputs)
    x = reshape1(x)
    x = main_net(x)
    x = reshape2(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model
