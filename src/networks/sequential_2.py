# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from networks.base import BaseNetwork

###

LEARNING_RATE = 0.01

###


class SequentialNetwork2(BaseNetwork):
    @property
    def uuid(self) -> str:
        return 'sequential-2'

    ###

    def build_model(self) -> Sequential:
        model = Sequential()

        model.add(self.input_layer())

        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(128, activation="relu"))

        model.add(self.output_layer(activation='relu'))

        print(model.summary())

        return model
