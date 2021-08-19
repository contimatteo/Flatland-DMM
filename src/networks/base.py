import abc

###


class BaseNetwork():
    def __init__(self, time_step_spec, action_spec) -> None:
        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

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

    @abc.abstractmethod
    def compile(self):
        raise NotImplementedError('`compile` method not implemented.')
