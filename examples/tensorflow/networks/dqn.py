"""
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
"""

import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory

import configs as Configs

###

INPUT_NODES = Configs.OBS_TREE_N_FEATURES
OUTPUT_NODES = 3

GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 5
LEARNING_RATE = 0.01

MEMORY_LIMIT = BATCH_SIZE * INPUT_NODES  # TODO: is this right?
MEMORY_WINDOW_LENGTH = 1  # DON'T TOUCH THIS

###


class NetworkDQN:
    @staticmethod
    def compile():
        model = Sequential()

        model.add(Dense(32, input_dim=INPUT_NODES, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(OUTPUT_NODES))

        model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))

        return model

    ###

    def __init__(self):
        self.env = None
        self.memory = None

        self.model = None
        self.target_model = None

    def __train(self):
        if self.memory.nb_entries < BATCH_SIZE + 2:
            return False

        # return number of {BATCH_SIZE} samples in random order.
        samples = self.memory.sample(BATCH_SIZE)

        for sample in samples:
            observation, action, reward, done, _ = sample
            observation = np.array(observation)

            target = self.target_model.predict(observation)

            if done is True:
                target[0][action] = reward
            else:
                q_future_value = max(self.target_model.predict(observation)[0])
                target[0][action] = reward + q_future_value * GAMMA

            self.model.fit([observation], target, epochs=1, verbose=0)

        return True

    def __target_model_weights_sync(self):
        self.target_model.set_weights(self.model.get_weights())

    ###

    def initialize(self, env):
        self.env = env
        self.memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=MEMORY_WINDOW_LENGTH)

        self.model = DQN.compile()
        self.target_model = DQN.compile()

    def remember(self, observation, action, reward, done, training=True):
        self.memory.append(observation, action, reward, done, training)

        trained = self.__train()

        if trained is True:
            self.__target_model_weights_sync()

    def predict(self, observation):
        return self.target_model.predict(observation)
