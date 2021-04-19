import json
import numpy as np

import configs as Configs

from agents.simple import SimpleAgent
from libs import logger
from libs.environment import Environment
from models.dqn import DQN

###

# TODO: renable this ...
# np.random.seed(Configs.RANDOM_SEED)

###

DEBUG = Configs.DEBUG

N_CELLS = Configs.RAIL_ENV_N_CELLS
N_AGENTS = Configs.NUMBER_OF_AGENTS
N_ATTEMPTS = Configs.TRAIN_N_ATTEMPTS
N_MAX_EPISODES = Configs.TRAIN_N_MAX_EPISODES

OBS_TREE_STATE_SIZE = Configs.OBSERVATION_TREE_STATE_SIZE

###


def train():
    attempt = 0

    # Flatland environemnt.
    environment = Environment()

    # Deep Learning model to use.
    model = DQN()
    model.initialize(env=environment)

    while attempt < N_ATTEMPTS:
        agents = []

        DEBUG and print("\n\n")
        DEBUG and logger.console.debug("ATTEMPT = {}".format(attempt))

        # instantiation of agents classes.
        for _ in range(N_AGENTS):
            random_agent = SimpleAgent()
            random_agent = random_agent.initialize(OBS_TREE_STATE_SIZE, model)
            agents.append(random_agent)

        score = 0
        done = None
        attempt += 1

        # get initial observation config.
        current_observations, info = environment.reset()

        for episode in range(N_MAX_EPISODES):
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

                current_obs = current_observations.get(i)
                next_obs = next_observations.get(i)

                weighted_reward = .1 if rewards[i] == -1 else (.5 if rewards[i] == 0 else 1)
                # weighted_reward = weighted_reward * (N_CELLS - current_obs.dist_min_to_target) / 10

                agents[i].step(current_obs, actions_taken[i], weighted_reward, next_obs, done[i])

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
