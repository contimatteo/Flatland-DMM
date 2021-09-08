from __future__ import absolute_import, division, print_function

from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from environments.keras import KerasEnvironment
from networks.sequential import SequentialNetwork
from observators.tree import BinaryTreeObservator

import configs as Configs

Adam._name = 'Adam'

###

N_AGENTS = Configs.TRAIN_N_AGENTS
N_ATTEMPTS = Configs.TRAIN_N_ATTEMPTS
N_EPISODES = Configs.TRAIN_N_EPISODES

MEMORY_LIMIT = 50000

LEARNING_RATE = 1e-3

target_model_update = 1e-2
N_ACTIONS = 3

###


class Runner():
    def __init__(self) -> None:
        pass

    #

    @staticmethod
    def build_callbacks(env_name):
        checkpoint_weights_filename = './dqn_' + env_name + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        return callbacks

    #

    def train(self) -> None:
        observator = BinaryTreeObservator(max_memory=Configs.OBS_TREE_N_NODES)
        env = KerasEnvironment(observator=observator)

        timestep_spec = env.time_step_spec()
        action_spec = env.action_spec()

        memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=1)
        network = SequentialNetwork(timestep_spec, action_spec)

        model = network.build_model()

        agent = DQNAgent(model, memory=memory, nb_actions=N_ACTIONS, target_model_update=1e-2)
        agent.compile(Adam(learning_rate=LEARNING_RATE), metrics=['mae'])

        # callbacks = Runner.build_callbacks(env_name='local')

        # agent.fit(env, nb_steps=1000, visualize=False, verbose=2, callbacks=callbacks)
        agent.fit(env, nb_steps=50000, visualize=False, verbose=2)
