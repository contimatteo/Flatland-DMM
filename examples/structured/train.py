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


def __compute_weighted_rewards(reward, observation):
    # weighted_reward = 1 if reward == 1 else (.5 if reward == 0 else .1)
    # weighted_reward *= (N_CELLS - observation.dist_min_to_target) / 2
    # return weighted_reward

    return reward


def __agent_is_done(done_map, agent_index):
    return done_map is not None and done_map.get(agent_index) is True


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
            weighted_rewards = {}

            # the selected agent will return the action to perform.
            for i in environment.get_agents_indexes():
                if __agent_is_done(done, i):
                    continue

                action = agents[i].act(current_observations.get(i), info, episode)
                actions_taken.update({i: action})

            # get the result of the performed action.
            next_observations, rewards, done, info = environment.perform_actions(actions_taken)

            # weight the rewards obtained.
            for i in environment.get_agents_indexes():
                if __agent_is_done(done, i):
                    continue

                weighted_rewards[i] = __compute_weighted_rewards(
                    rewards[i], current_observations.get(i)
                )

            DEBUG and logger.console.debug(
                "episode = {:0>3d}      actions = {}      rewards = {}".
                format(episode, str(actions_taken), json.dumps(weighted_rewards))
            )

            # train the agent by giving action taken and the result.
            for i in environment.get_agents_indexes():
                if __agent_is_done(done, i):
                    continue

                score += rewards[i]

                agents[i].step(
                    current_observations.get(i), actions_taken[i], weighted_rewards[i],
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
