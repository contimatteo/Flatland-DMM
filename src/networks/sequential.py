from keras.layers import Dense
from keras import Sequential
from keras.optimizers import adam_v2

from networks.base import BaseNetwork

###

LEARNING_RATE = 0.01

###


class SequentialNetwork(BaseNetwork):
    @property
    def input_nodes(self) -> int:
        return 0

    @property
    def input_dim(self) -> int:
        return 0

    @property
    def output_nodes(self) -> int:
        return 0

    ###

    def compile(self) -> Sequential:
        model = Sequential()

        model.add(Dense(self.input_nodes, input_dim=self.input_dim, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.output_nodes))

        model.compile(optimizer=adam_v2.Adam(lr=LEARNING_RATE))

        return model
