"""
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
"""

from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam
from rl.memory import SequentialMemory

###

GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 32
LEARNING_RATE = 0.01

MEMORY_LIMIT = 50000
MEMORY_WINDOW_LENGTH = 1

###


class DQN:
    @staticmethod
    def compile_model(input_nodes, output_nodes):
        model = Sequential()

        model.add(Dense(24, input_dim=input_nodes, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(output_nodes))

        model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))

        return model

    ###

    def __init__(self):
        self.env = None
        self.action_space = None
        self.observation_space = None
        self.memory = None
        self.model = None
        self.target_model = None

    def initialize(self, env, action_space, observation_space):
        self.env = env
        self.action_space = action_space
        self.observation_space = observation_space

        self.memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=MEMORY_WINDOW_LENGTH)

        input_nodes = self.observation_space.shape[0]  # TODO: check this
        output_nodes = self.action_space  # TODO: check this

        self.model = DQN.compile_model(input_nodes, output_nodes)
        self.target_model = DQN.compile_model(input_nodes, output_nodes)

    def remember(self, observation, action, reward, new_state, done, training=True):
        state = (observation, new_state)
        self.memory.append(state, action, reward, done, training)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        samples = self.memory.sample(BATCH_SIZE)

        for sample in samples:
            state, action, reward, done = sample
            (observation, new_state) = state

            target = self.target_model.predict(state)

            if done:
                target[0][action] = reward
            else:
                q_future_value = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q_future_value * GAMMA

            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for (i, _) in enumerate(target_weights):
            target_weights[i] = weights[i]

        self.target_model.set_weights(target_weights)
