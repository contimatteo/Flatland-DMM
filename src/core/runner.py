from __future__ import absolute_import, division, print_function

import configs as Configs

from core import prepare_env, prepare_memory, prepare_network, prepare_policy, prepare_callbacks
from marl.dqn import DQNMultiAgent

###


class Runner():
    def __init__(self) -> None:
        pass

    #

    def train(self) -> None:
        visualize = False
        nb_steps = Configs.TRAIN_N_STEPS
        verbose = Configs.DQN_AGENT_VERBOSE

        env = prepare_env()
        memory = prepare_memory()
        policy = prepare_policy()
        callbacks = prepare_callbacks()
        network, optimizer, metrics = prepare_network(env)

        agent = DQNMultiAgent(
            policy=policy,
            memory=memory,
            model=network.keras_model,
            nb_actions=Configs.N_ACTIONS,
            target_model_update=Configs.DQN_AGENT_TARGET_MODEL_UPDATE,
        )

        agent.compile(optimizer, metrics=metrics)

        for _ in range(Configs.TRAIN_N_ATTEMPTS):
            agent.fit(env, nb_steps, visualize=visualize, callbacks=callbacks, verbose=verbose)
