import json
import numpy as np

import configs as Configs

from agents.random import RandomAgent
from libs import logger
from libs.environment import Environment
from models.sequential import SequentialModel

###

np.random.seed(Configs.RANDOM_SEED)

###


def train():
    attempt = 0
    agents = list()

    # (2^max_depth * tree_obs_features) + 1
    STATE_SIZE = (2**(Configs.OBSERVATOR_TREE_MAX_DEPTH * 9)) + 1

    # Deep Learning model to use.
    model = SequentialModel()
    model = model.initialize()

    # Flatland environemnt.
    environment = Environment()

    # instantiation of agents classes.
    for _ in range(Configs.NUMBER_OF_AGENTS):
        random_agent = RandomAgent()
        random_agent = random_agent.initialize(STATE_SIZE, model)

        agents.append(random_agent)

    while attempt < Configs.TRAIN_N_MAX_ATTEMPTS:
        score = 0
        done = None
        attempt += 1

        # get initial observation config.
        current_observations, info = environment.reset()

        for _ in range(Configs.TRAIN_N_EPISODES):
            actions_taken = dict()

            # the selected agent will return the action to perform.
            for i in environment.get_agents_indexes():
                action = agents[i].act(current_observations.get(i), info)
                actions_taken.update({i: action})

            # get the result of the performed action.
            next_observations, rewards, done, info = environment.perform_actions(actions_taken)

            # train the agent by giving action taken and the result.
            for i in environment.get_agents_indexes():
                score += rewards[i]

                agents[i].step(current_observations.get(i), actions_taken[i], rewards[i],
                               next_observations.get(i), done[i])

            # save obesrvations for next iteration.
            current_observations = next_observations.copy()

            # refresh emulator window
            environment.render(sleep_seconds=.4)

            if environment.finished(done):
                break

        logger.console.debug("score = %s", score)


###

if __name__ == '__main__':
    train()
