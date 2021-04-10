import numpy as np
import json

import configs as Configs

from agents.random import Agent
from libs import logger
from libs.environment import Environment

###

np.random.seed(Configs.RANDOM_SEED)

###


def main():
    attempt = 0
    actions = dict()
    STATE_SIZE = 218  # TODO: automatically compute this.

    environment = Environment()
    agent = Agent(218)  # (2^max_depth * tree_obs_features) + 1

    while attempt < Configs.TRAIN_N_MAX_ATTEMPTS:
        score = 0
        done = None
        attempt += 1

        observations, info = environment.reset()

        print("==================================================================================")
        logger.console.debug("attempt = {}".format(attempt))
        logger.console.debug("initial-observation = {}".format(json.dumps(observations, indent=2)))
        logger.console.debug("initial-info = {}".format(json.dumps(info, indent=2)))
        # input('press Enter to start ...')

        for _ in range(Configs.TRAIN_N_EPISODES):
            for i in environment.get_agents_indexes():
                action = agent.act(observations[i], info)
                actions.update({i: action})

            next_observations, rewards, done, info = environment.perform_actions(actions)

            for i in environment.get_agents_indexes():
                score += rewards[i]
                agent.step((observations[i], actions[i], rewards[i], next_observations[i], done[i]))

            observations = next_observations.copy()
            environment.render(sleep_seconds=.4)

            if environment.finished():
                break

        logger.console.debug("score = {}".format(score))

        if environment.finished(done):
            break


###

if __name__ == '__main__':
    main()
