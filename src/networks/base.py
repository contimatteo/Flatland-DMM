import abc

from tensorflow.keras import Sequential

###


class BaseNetwork(abc.ABC):
    def __init__(self, time_step_spec, action_spec) -> None:
        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self._keras_model = self.build_model()

    ###

    @abc.abstractproperty
    def input_nodes(self) -> int:
        raise NotImplementedError('`input_nodes` property not implemented.')

    @abc.abstractproperty
    def input_dim(self) -> int:
        raise NotImplementedError('`input_dim` property not implemented.')

    @abc.abstractproperty
    def output_nodes(self) -> int:
        raise NotImplementedError('`output_nodes` property not implemented.')

    # @abc.abstractmethod
    # def compile(self) -> Model:
    #     raise NotImplementedError('`compile` method not implemented.')

    @abc.abstractmethod
    def build_model(self) -> Sequential:
        raise NotImplementedError('`_init_model` method not implemented.')
