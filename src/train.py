from typing import List

import numpy as np

from agents.random import RandomAgent
from environments.py_env import PyEnvironment
from models.dqn import DQN
from networks.sequential import SequentialNetwork
from observators.tree import BinaryTreeObservator

import configs as Configs

###

N_AGENTS = Configs.N_OF_AGENTS
N_ATTEMPTS = Configs.TRAIN_N_ATTEMPTS
N_EPISODES = Configs.TRAIN_N_EPISODES

###

np.random.seed(Configs.RANDOM_SEED)

###


def prepare_env():
    observator = BinaryTreeObservator()
    environment = PyEnvironment(observator=observator)

    return environment


def prepare_model(environment):
    time_step_spec = environment.time_step_spec()
    action_spec = environment.action_spec()

    model = DQN(SequentialNetwork, time_step_spec, action_spec)

    return model


def prepare_agents(environment) -> List[RandomAgent]:
    agents: List[RandomAgent] = []

    # configure the DL model to use.
    model = prepare_model(environment)

    time_step_spec = environment.time_step_spec()
    action_spec = environment.action_spec()

    for _ in range(N_AGENTS):
        agent = RandomAgent(model, time_step_spec, action_spec)
        agents.append(agent)

    return agents


def train():
    environment = prepare_env()

    time_step = environment.reset()

    for _ in range(N_ATTEMPTS):
        agents = prepare_agents(environment)

        for _ in range(N_EPISODES):
            actions = {}

            # perform actions
            for i in range(N_AGENTS):
                actions.update({i: agents[i].act(time_step)})

            # get the new observations given the actions
            time_step = environment.step(actions)

            # share with the agents the reward obtained
            for i in range(N_AGENTS):
                agents[i].step(actions[i], time_step)

            # check if all agents have reached the goal
            if time_step.is_last().all():
                time_step = environment.reset()


###

if __name__ == '__main__':
    train()
