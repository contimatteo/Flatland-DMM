from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

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

    @property
    def keras_model(self) -> int:
        return self._keras_model

    ###

    def build_model(self) -> Sequential:
        model = Sequential()

        model.add(Flatten(input_shape=(1,) + (self.input_nodes,)))
        model.add(Dense(self.input_nodes, input_dim=self.input_dim, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.output_nodes))

        return model

    # def compile(self) -> Sequential:
    #     model = Sequential()

    #     model.add(Dense(self.input_nodes, input_dim=self.input_dim, activation="relu"))
    #     model.add(Dense(10, activation="relu"))
    #     model.add(Dense(self.output_nodes))

    #     model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=mean_squared_error)

    #     return model
