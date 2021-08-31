from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

from networks.base import BaseNetwork

###

LEARNING_RATE = 0.01

###


class SequentialNetwork(BaseNetwork):
    @property
    def input_nodes(self) -> int:
        return self._time_step_spec.observation.shape[0]

    @property
    def input_dim(self) -> int:
        if len(self._time_step_spec.observation.shape) > 1:
            return self._time_step_spec.observation.shape[1]
        return 1

    @property
    def output_nodes(self) -> int:
        return (self._action_spec.maximum - self._action_spec.minimum) + 1

    ###

    def compile(self) -> Sequential:
        model = Sequential()

        model.add(Dense(self.input_nodes, input_dim=self.input_dim, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.output_nodes))

        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=mean_squared_error)

        return model
