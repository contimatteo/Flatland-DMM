from __future__ import absolute_import, division, print_function

from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam

from environments.prepare_env import prepare_env
from marl.callbacks import FileLogger, ModelIntervalCheckpoint
from marl.dqn import DQNMultiAgent
from networks.sequential import SequentialNetwork

import configs as Configs

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
        observations_shape = (39,)
        n_actions = 3

        env = prepare_env()
        network = SequentialNetwork(observations_shape, n_actions)
        memory = SequentialMemory(limit=Configs.DQN_AGENT_MEMORY_LIMIT, window_length=1)

        agent = DQNMultiAgent(
            memory=memory,
            model=network.keras_model,
            nb_actions=Configs.N_ACTIONS,
            target_model_update=Configs.DQN_AGENT_TARGET_MODEL_UPDATE,
        )

        agent.compile(Adam(learning_rate=Configs.DQN_AGENT_LEARNING_RATE), metrics=['mae'])

        # callbacks = Runner.build_callbacks(env_name='local')
        # agent.fit(env, nb_steps=1000, visualize=False, verbose=2, callbacks=callbacks)

        agent.fit(
            env,
            visualize=False,
            nb_steps=Configs.TRAIN_N_STEPS,
            verbose=Configs.DQN_AGENT_VERBOSE,
            nb_max_episode_steps=Configs.TRAIN_N_MAX_EPISODE_STEPS,
        )
