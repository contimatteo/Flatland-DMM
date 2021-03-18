import numpy as np
import json
import logging

import configs as Configs

from agents.random import Agent
from utils import logger
from utils.rail_env import Environment

###

np.random.seed(Configs.RANDOM_SEED)

###


def train():
    done = None
    attempt = -1
    actions = dict()
    STATE_SIZE = 218 # TODO: automatically compute this.

    environment = Environment()
    agent = Agent(218)

    while not environment.finished(done) and attempt < Configs.TRAIN_N_MAX_ATTEMPTS:
        score = 0
        attempt += 1

        observations, info = environment.reset()

        for _ in range(Configs.TRAIN_N_EPISODES):
            for i in environment.get_agents_indexes():
                action = agent.act(observations[i], info)
                actions.update({i: action})

            next_observations, rewards, done, info = environment.perform_actions(actions)

            for i in environment.get_agents_indexes():
                score += rewards[i]
                agent.step((observations[i], actions[i], rewards[i], next_observations[i], done[i]))

            observations = next_observations.copy()

            environment.render()

            if environment.finished():
                break

        logger.console.debug("Attempt Nr. {}\t Score = {}".format(attempt + 1, score))


###

if __name__ == '__main__':
    train()
