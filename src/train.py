from typing import List
from dotenv import load_dotenv

import numpy as np

from agents.random import RandomAgent
from environments.py_env import PyEnvironment
from models.dqn import DQN
from networks.sequential import SequentialNetwork
from observators.tree import BinaryTreeObservator
from utils import logger

import configs as Configs

###

DEBUG = Configs.DEBUG

N_AGENTS = Configs.N_OF_AGENTS
N_ATTEMPTS = Configs.TRAIN_N_ATTEMPTS
N_EPISODES = Configs.TRAIN_N_EPISODES

###

load_dotenv()

np.random.seed(Configs.RANDOM_SEED)

###


def prepare_env() -> PyEnvironment:
    observator = BinaryTreeObservator(max_memory=Configs.OBS_TREE_N_NODES)
    environment = PyEnvironment(observator=observator)

    return environment


def prepare_model(environment) -> DQN:
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

    for attempt in range(N_ATTEMPTS):
        DEBUG and print("\n\n")
        DEBUG and print("========================================================================")
        DEBUG and print("========================================================================")
        DEBUG and print("\n\n")

        # prepare the environment
        time_step = environment.reset()

        # prepare the agents
        agents = prepare_agents(environment)

        for episode in range(N_EPISODES):
            DEBUG and logger.console.debug(
                "Attempt {}/{}  |  Episode {}/{}".
                format(attempt + 1, N_ATTEMPTS, episode + 1, N_EPISODES)
            )

            # perform actions
            actions = {}
            for i in range(N_AGENTS):
                # TODO: check if only one action is allowed.
                # ...

                actions.update({i: agents[i].act(time_step)})

            # get the new observations given the actions
            time_steps_dict = environment.step(actions)

            # check if all agents have reached the goal
            if environment.is_episode_finished() is True:
                break

            # share with the agents the reward obtained
            for i in time_steps_dict.keys():
                agents[i].step(actions[i], time_steps_dict[i], environment.get_done()[i])

        DEBUG and print("\n\n")


###

if __name__ == '__main__':
    train()
