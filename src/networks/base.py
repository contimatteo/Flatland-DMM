from typing import Tuple
import abc

from tensorflow.keras import Sequential

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

    def load_weights(self):
        pass 

    ###

    @abc.abstractproperty
    def name(self) -> str:
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
