from __future__ import absolute_import, division, print_function

import configs as Configs

from core import prepare_env, prepare_memory, prepare_network, prepare_policy, prepare_callbacks
from marl.dqn import DQNMultiAgent

###


class Runner():
    def __init__(self) -> None:
        pass

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
        )

        agent.compile(optimizer, metrics=metrics)

        return agent

    #

    def train(self) -> None:
        visualize = False
        nb_steps = Configs.TRAIN_N_STEPS
        verbose = Configs.DQN_AGENT_TRAIN_VERBOSE

        env = prepare_env()
        callbacks = prepare_callbacks()

        agent = self._prepare_agent(env)

        for attempt in range(Configs.TRAIN_N_ATTEMPTS):
            print()
            print("##############################################################################")
            print(f"ATTEMPT {attempt+1}/{Configs.TRAIN_N_ATTEMPTS}")
            agent.fit(env, nb_steps, visualize=visualize, callbacks=callbacks, verbose=verbose)

    def test(self):
        visualize = False
        nb_episodes = Configs.TEST_N_ATTEMPTS
        verbose = Configs.DQN_AGENT_TEST_VERBOSE

        env = prepare_env()
        callbacks = prepare_callbacks()

        agent = self._prepare_agent(env)

        agent.test(
            env, nb_episodes=nb_episodes, visualize=visualize, callbacks=callbacks, verbose=verbose
        )
