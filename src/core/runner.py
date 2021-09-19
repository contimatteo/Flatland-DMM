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
from marl.dqn import DQNMultiAgent
from utils.storage import Storage

###

warnings.filterwarnings('ignore')

load_dotenv()

if Configs.APP_SEED is not None:
    np.random.seed(Configs.APP_SEED)

###


class Runner():
    def __init__(self) -> None:
        Storage.initialize()

    #

    def _prepare_agent(self, env):
        nb_actions = Configs.N_ACTIONS

        env = prepare_env()
        memory = prepare_memory()
        policy = prepare_policy(Configs.POLICY_TYPE, Configs.POLICY_PARAMETERS)
        network, optimizer, metrics = prepare_network(env)

        agent = DQNMultiAgent(
            policy=policy,
            memory=memory,
            nb_actions=nb_actions,
            model=network.keras_model,
            target_model_update=Configs.DQN_AGENT_TARGET_MODEL_UPDATE,
            nb_steps_warmup=Configs.TRAIN_N_STEPS_WARMUP,
        )

        agent.compile(optimizer, metrics=metrics)

        return agent, network

    #

    def train(self) -> None:
        visualize = False
        nb_steps = Configs.TRAIN_N_STEPS
        verbose = Configs.DQN_AGENT_TRAIN_VERBOSE
        max_episode_steps = Configs.TRAIN_N_MAX_STEPS_FOR_EPISODE
        # log_interval = Configs.TRAIN_N_STEPS_WARMUP * Configs.N_AGENTS
        log_interval = Configs.TRAIN_LOG_INTERVAL

        env = prepare_env()
        agent, network = self._prepare_agent(env)
        callbacks = prepare_callbacks([], network)

        if Path(network.weights_file_url).is_file() is True:
            agent.load_weights(network.weights_file_url)

        agent.fit(
            env,
            nb_steps,
            verbose=verbose,
            visualize=visualize,
            callbacks=callbacks,
            nb_max_episode_steps=max_episode_steps,
            log_interval=log_interval,
        )

        agent.save_weights(network.weights_file_url, overwrite=True)

    def test(self):
        visualize = False
        nb_episodes = Configs.TEST_N_ATTEMPTS
        verbose = Configs.DQN_AGENT_TEST_VERBOSE
        max_episode_steps = Configs.TEST_N_MAX_STEPS_FOR_EPISODE

        env = prepare_env()
        agent, network = self._prepare_agent(env)
        callbacks = prepare_callbacks([], network)

        assert Path(network.weights_file_url).is_file() is True
        agent.load_weights(network.weights_file_url)

        agent.test(
            env,
            verbose=verbose,
            visualize=visualize,
            callbacks=callbacks,
            nb_episodes=nb_episodes,
            nb_max_episode_steps=max_episode_steps,
        )
