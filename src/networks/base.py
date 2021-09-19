from typing import Tuple
import abc

from tensorflow.keras import Sequential

from utils import Storage

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

    @abc.abstractproperty
    def uuid(self) -> str:
        raise NotImplementedError('`name` property not implemented.')

    @abc.abstractproperty
    def input_nodes(self) -> int:
        raise NotImplementedError('`input_nodes` property not implemented.')

    @abc.abstractproperty
    def input_dim(self) -> int:
        raise NotImplementedError('`input_dim` property not implemented.')

    @abc.abstractproperty
    def output_nodes(self) -> int:
        raise NotImplementedError('`output_nodes` property not implemented.')

    @abc.abstractmethod
    def build_model(self) -> Sequential:
        raise NotImplementedError('`_init_model` method not implemented.')
