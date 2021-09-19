from __future__ import absolute_import, division, print_function
from pathlib import Path

import configs as Configs

from core import prepare_env, prepare_memory, prepare_network, prepare_policy, prepare_callbacks
from marl.dqn import DQNMultiAgent
from utils import Storage

###


class Runner():
    def __init__(self) -> None:
        Storage.initialize()

    #

    def _prepare_agent(self, env):
        nb_actions = Configs.N_ACTIONS

        env = prepare_env()
        memory = prepare_memory()
        policy = prepare_policy()
        network, optimizer, metrics = prepare_network(env)

        agent = DQNMultiAgent(
            policy=policy,
            memory=memory,
            nb_actions=nb_actions,
            model=network.keras_model,
            target_model_update=Configs.DQN_AGENT_TARGET_MODEL_UPDATE,
            nb_steps_warmup=100,
        )

        agent.compile(optimizer, metrics=metrics)

        return agent, network

    #

    def train(self) -> None:
        visualize = False
        nb_steps = Configs.TRAIN_N_STEPS
        verbose = Configs.DQN_AGENT_TRAIN_VERBOSE
        max_episode_steps = Configs.TRAIN_N_MAX_STEPS_FOR_EPISODE

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
        )

        agent.save_weights(network.weights_file_url)

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
