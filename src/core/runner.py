from __future__ import absolute_import, division, print_function

from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.optimizers import Adam
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

import configs as Configs

from environments.prepare_env import prepare_env
from marl.dqn import DQNMultiAgent
from networks.sequential import SequentialNetwork

###


class Runner():
    def __init__(self) -> None:
        pass

    #

    @staticmethod
    def build_callbacks(env_name):
        checkpoint_weights_filename = './tmp/dqn_' + env_name + '_weights_{step}.h5f'
        log_filename = 'tmp/dqn_{}_log.json'.format(env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500)]
        callbacks += [FileLogger(log_filename, interval=100)]
        return callbacks

    #

    def train(self) -> None:
        env = prepare_env()

        observations_shape = env.observation_space.shape
        n_actions = env.action_space.n

        network = SequentialNetwork(observations_shape, n_actions)
        memory = SequentialMemory(limit=Configs.DQN_AGENT_MEMORY_LIMIT, window_length=1)

        # policy = BoltzmannQPolicy()
        callbacks = Runner.build_callbacks(env_name='local')

        agent = DQNMultiAgent(
            # policy=policy,
            memory=memory,
            model=network.keras_model,
            nb_actions=n_actions,
            target_model_update=Configs.DQN_AGENT_TARGET_MODEL_UPDATE,
        )

        agent.compile(Adam(learning_rate=Configs.DQN_AGENT_LEARNING_RATE), metrics=['mae'])

        for _ in range(Configs.TRAIN_N_ATTEMPTS):
            agent.fit(
                env,
                # callbacks=callbacks,
                nb_steps=Configs.TRAIN_N_STEPS,
                verbose=Configs.DQN_AGENT_VERBOSE,
            )
