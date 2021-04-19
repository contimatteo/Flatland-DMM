"""
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
"""

from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam
from rl.memory import SequentialMemory

import configs as Configs

###

ACTIONS_SPACE_SIZE = 5
OBS_TREE_STATE_SIZE = Configs.OBSERVATION_TREE_STATE_SIZE

GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 25
LEARNING_RATE = 0.01

MEMORY_LIMIT = OBS_TREE_STATE_SIZE * BATCH_SIZE  # TODO: is this right?

###


class DQN:
    @staticmethod
    def compile_model(input_nodes, output_nodes):
        model = Sequential()

        model.add(Dense(64, input_dim=input_nodes, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(output_nodes))

        model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))

        return model

    ###

    def __init__(self):
        self.env = None
        self.action_space = ACTIONS_SPACE_SIZE
        self.observation_space = OBS_TREE_STATE_SIZE

        self.memory = None

        self.model = None
        self.target_model = None
        self.model_weights_updated = False

    def __train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # return number of {BATCH_SIZE} samples in random order.
        samples = self.memory.sample(BATCH_SIZE)

        if len(samples) > 0:
            self.model_weights_updated = True

        for sample in samples:
            observation, action, reward, done, _ = sample

            target = self.target_model.predict(observation)

            if done:
                target[0][action] = reward
            else:
                q_future_value = max(self.target_model.predict(observation)[0])
                target[0][action] = reward + q_future_value * GAMMA

            self.model.fit(observation, target, epochs=1, verbose=0)

    def __target_model_weights_sync(self):
        if self.model_weights_updated is not True:
            return

        self.model_weights_updated = False

        self.target_model.set_weights(self.model.get_weights())

        # weights = self.model.get_weights()
        # target_weights = self.target_model.get_weights()
        # for (i, _) in enumerate(target_weights):
        #     target_weights[i] = weights[i]
        # self.target_model.set_weights(target_weights)

    ###

    def initialize(self, env):
        self.env = env
        self.memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=BATCH_SIZE)

        input_nodes = self.observation_space
        output_nodes = self.action_space

        self.model = DQN.compile_model(input_nodes, output_nodes)
        self.target_model = DQN.compile_model(input_nodes, output_nodes)

    def remember(self, observation, action, reward, done, training=True):
        self.memory.append(observation, action, reward, done, training)

        self.__train()

        self.__target_model_weights_sync()

    def predict(self, next_obs):
        return self.target_model.predict(next_obs)
