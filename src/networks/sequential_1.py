# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from networks.base import BaseNetwork

###

LEARNING_RATE = 0.01

###


class SequentialNetwork1(BaseNetwork):
    @property
    def uuid(self) -> str:
        return 'sequential-1'

    ###

    def build_model(self) -> Sequential:
        model = Sequential()

        # model.add(Flatten(input_shape=(1, self.input_nodes)))
        model.add(self.input_layer())

        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))

        model.add(self.output_layer())
        # model.add(Dense(self.output_nodes, activation="linear"))

        print(model.summary())

        return model
