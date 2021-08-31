import abc

from rl.memory import SequentialMemory

###

MEMORY_LIMIT = 100
MEMORY_WINDOW_LENGTH = 1  # TODO: is this right?

###


class BaseModel(abc.ABC):
    def __init__(self, NetworkClass, time_step_spec, action_spec):
        self.memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=MEMORY_WINDOW_LENGTH)

        # TODO: find a way to pass only the Keras instance and not the entire {NetworkClass}
        self.network = NetworkClass(time_step_spec, action_spec).compile()
        self.target_network = NetworkClass(time_step_spec, action_spec).compile()

    ###

    def remember(self, action, observation, reward: float, finished: bool, training=True):
        observation_dict = { 'network_input': observation }
        self.memory.append(observation_dict, action, reward, finished, training)

        trained = self._train()

        if trained is True:
            self.__sync_target_network()

    def predict(self, observation):
        return self.target_network.predict(observation)

    def __sync_target_network(self):
        self.target_network.set_weights(self.network.get_weights())

    ###

    @abc.abstractmethod
    def _train(self):
        raise NotImplementedError('`_train` method not implemented.')
