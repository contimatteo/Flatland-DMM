import numpy as np
import json

import configs as Configs

from agents.random import Agent
from utils.rail_env import Environment

###

np.random.seed(Configs.RANDOM_SEED)

###


def main():
    # TODO: automatically compute this.
    STATE_SIZE = 218

    environment = Environment()
    agent = Agent(218, Configs.ACTIONS_SIZE)

    attempt = -1
    actions = dict()
    done = None

    while not environment.finished(done) or attempt < Configs.TRAIN_N_MAX_ATTEMPTS:
        score = 0
        attempt += 1
        observations, info = environment.reset()

        for _ in range(Configs.TRAIN_N_EPISODES):
            for i in range(Configs.NUMBER_OF_AGENTS):
                action = agent.act(observations[i], info)
                actions.update({i: action})

            next_observations, rewards, done, _ = environment.perform_actions(actions)

            for i in range(Configs.NUMBER_OF_AGENTS):
                o = observations[i]
                a = actions[i]
                r = rewards[i]
                no = next_observations[i]
                d = done[i]

                agent.step((o, a, r, no, d))
                score += rewards[i]

            observations = next_observations.copy()

            environment.render()
            environment.sleep(.05)

            if environment.finished():
                break

        print('Episode Nr. {}\t Score = {}'.format(attempt + 1, score))


###

if __name__ == '__main__':
    main()
