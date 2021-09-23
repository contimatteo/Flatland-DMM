from __future__ import absolute_import, division, print_function
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import warnings

from configs import configurator as Configs
from core import prepare_env
from core import prepare_memory
from core import prepare_network
from core import prepare_policy
from core import prepare_callbacks
from core import prepare_optimizer
from core import prepare_metrics
from marl.dqn import DQNMultiAgent
from utils.storage import Storage

###

warnings.filterwarnings('ignore')

load_dotenv()

###


class Runner():
    def __init__(self) -> None:
        Storage.initialize()

    #

    def _prepare_agent(self, env):
        env = prepare_env()
        policy = prepare_policy()
        memory = prepare_memory()
        metrics = prepare_metrics()
        network = prepare_network(env)
        optimizer = prepare_optimizer()

        agent = DQNMultiAgent(
            policy=policy,
            memory=memory,
            nb_actions=Configs.N_ACTIONS,
            model=network.keras_model,
            **Configs.AGENT_PARAMS,
        )

        agent.compile(optimizer, metrics=metrics)

        return agent, network

    #

    def train(self) -> None:
        env = prepare_env()
        agent, network = self._prepare_agent(env)
        callbacks = prepare_callbacks(training=True)

        if Path(network.weights_file_url).is_file() is True:
            agent.load_weights(network.weights_file_url)

        agent.fit(
            env,
            Configs.TRAIN_N_STEPS,
            visualize=False,
            callbacks=callbacks,
            verbose=Configs.TRAIN_VERBOSE,
            log_interval=Configs.TRAIN_LOG_INTERVAL,
            nb_max_episode_steps=Configs.TRAIN_N_MAX_STEPS_FOR_EPISODE,
        )

        agent.save_weights(network.weights_file_url, overwrite=True)

    def test(self):
        env = prepare_env()
        agent, network = self._prepare_agent(env)
        callbacks = prepare_callbacks(training=False)

        assert Path(network.weights_file_url).is_file() is True
        agent.load_weights(network.weights_file_url)

        agent.test(
            env,
            visualize=False,
            callbacks=callbacks,
            verbose=Configs.TEST_VERBOSE,
            nb_episodes=Configs.TEST_N_ATTEMPTS,
            nb_max_episode_steps=Configs.TEST_N_MAX_STEPS_FOR_EPISODE,
        )
