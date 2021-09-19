from typing import Tuple
import abc

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Sequential

from utils.storage import Storage

###


class BaseNetwork(abc.ABC):
    def __init__(self, observations_shape: Tuple, n_actions: int) -> None:
        self._observations_shape = observations_shape
        self._n_actions = n_actions

        self._keras_model = self.build_model()

    ###

    @property
    def keras_model(self) -> int:
        return self._keras_model

    @property
    def _weights_file_name(self) -> str:
        return f"{self.uuid}.h5"

    @property
    def _weights_intervals_file_name(self) -> str:
        return self.uuid + "-{step}.h5"

    @property
    def weights_file_url(self) -> str:
        file_name = self._weights_file_name
        return str(Storage.weights_folder().joinpath(file_name).absolute())

    @property
    def weights_intervals_file_url(self) -> str:
        file_name = self._weights_intervals_file_name
        return str(Storage.weights_intervals_folder().joinpath(file_name).absolute())

    ###

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

    def input_layer(self) -> Flatten:
        return Flatten(input_shape=(1, self.input_nodes))

    def output_layer(self, activation="linear") -> Dense:
        return Dense(self.output_nodes, activation=activation)

    #

    @abc.abstractproperty
    def uuid(self) -> str:
        raise NotImplementedError('`uuid` property not implemented.')

    @abc.abstractmethod
    def build_model(self) -> Sequential:
        raise NotImplementedError('`_init_model` method not implemented.')
