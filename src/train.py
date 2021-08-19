import numpy as np

from tf_agents.policies import random_py_policy

import configs as Configs

from environments.py_env import PyEnvironment
from observators.tree import BinaryTreeObservator

###

np.random.seed(Configs.RANDOM_SEED)

###


def train():
    observator = BinaryTreeObservator()
    environment = PyEnvironment(observator=observator)

    time_step_spec = environment.time_step_spec()
    action_spec = environment.action_spec()

    policy = random_py_policy.RandomPyPolicy(time_step_spec=time_step_spec, action_spec=action_spec)

    episode_count = 0
    time_step = environment.reset()

    while episode_count < 100:
        action = policy.action(time_step).action

        time_step = environment.step({0: action, 1: action})

        if time_step.is_last().all():
            episode_count += 1
            time_step = environment.reset()


###

if __name__ == '__main__':
    train()
