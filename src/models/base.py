import abc

from rl.memory import SequentialMemory

###

MEMORY_WINDOW_LENGTH = 1  # TODO: is this right?

BATCH_SIZE = 5

###


class BaseModel():
    def __init__(self, NetworkClass, time_step_spec, action_spec):
        memory_limit = BATCH_SIZE * time_step_spec.observations.shape[0]

        self.memory = SequentialMemory(limit=memory_limit, window_length=MEMORY_WINDOW_LENGTH)

        self.network = NetworkClass(time_step_spec, action_spec).compile()
        self.target_network = NetworkClass(time_step_spec, action_spec).compile()

    ###

    def remember(self, observation, action, reward, done, training=True):
        self.memory.append(observation, action, reward, done, training)

        trained = self.__train()

        if trained is True:
            self.__sync_target_network()

    def predict(self, observation):
        return self.target_network.predict(observation)

    def __sync_target_network(self):
        self.target_network.set_weights(self.network.get_weights())

    ###

    @abc.abstractmethod
    def __train(self):
        raise NotImplementedError('`_train` method not implemented.')
