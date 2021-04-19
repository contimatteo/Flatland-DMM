import json
import numpy as np

import configs as Configs

from agents.simple import SimpleAgent
from libs import logger
from libs.environment import Environment
from models.sequential import SequentialModel
from models.dqn import DQN

###

DEBUG = Configs.DEBUG

# TODO: renable this ...
# np.random.seed(Configs.RANDOM_SEED)

###


def train():
    attempt = 0

    # (2^max_depth * tree_obs_features) + 1
    STATE_SIZE = (2**(Configs.OBSERVATOR_TREE_MAX_DEPTH * 9)) + 1

    # Flatland environemnt.
    environment = Environment()

    # Deep Learning model to use.
    model = SequentialModel()
    model = model.initialize()

    while attempt < Configs.TRAIN_N_ATTEMPTS:
        agents = []

        DEBUG and print("\n\n")
        DEBUG and logger.console.debug("ATTEMPT = {}".format(attempt))

        # instantiation of agents classes.
        for _ in range(Configs.NUMBER_OF_AGENTS):
            random_agent = SimpleAgent()
            random_agent = random_agent.initialize(STATE_SIZE, model)
            agents.append(random_agent)

        score = 0
        done = None
        attempt += 1

        # get initial observation config.
        current_observations, info = environment.reset()

        for episode in range(Configs.TRAIN_N_MAX_EPISODES):
            actions_taken = dict()

            # the selected agent will return the action to perform.
            for i in environment.get_agents_indexes():
                action = agents[i].act(current_observations.get(i), info)
                actions_taken.update({i: action})

            # get the result of the performed action.
            next_observations, rewards, done, info = environment.perform_actions(actions_taken)

            DEBUG and logger.console.debug(
                "episode = {:0>3d}      actions = {}      rewards = {}".
                format(episode, str(actions_taken), json.dumps(rewards))
            )

            # train the agent by giving action taken and the result.
            for i in environment.get_agents_indexes():
                score += rewards[i]  # * current_observations.get(i).dist_min_to_target

                agents[i].step(
                    current_observations.get(i), actions_taken[i], rewards[i],
                    next_observations.get(i), done[i]
                )

            # save obesrvations for next iteration.
            current_observations = next_observations.copy()

            # refresh emulator window
            environment.render(sleep_seconds=.4)

            if environment.finished(done):
                break

        DEBUG and logger.console.debug("FINAL SCORE = {}".format(score))
        DEBUG and print("\n\n")


###

if __name__ == '__main__':
    train()
