from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential

from networks.base import BaseNetwork

###

LEARNING_RATE = 0.01

###


class SequentialNetwork1(BaseNetwork):
    @property
    def uuid(self) -> str:
        return 'sequential-1'

    @property
    def input_nodes(self) -> int:
        return self._observations_shape[0]

    @property
    def input_dim(self) -> int:
        if len(self._observations_shape) > 1:
            return self._observations_shape[1]
        return 1

    @property
    def output_nodes(self) -> int:
        return self._n_actions

    ###

    def build_model(self) -> Sequential:
        model = Sequential()

        model.add(Flatten(input_shape=(1, self.input_nodes)))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(self.output_nodes, activation="linear"))

        print(model.summary())

        return model
