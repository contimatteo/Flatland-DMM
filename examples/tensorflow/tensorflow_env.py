from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

import configs as Configs

from environments.pyenv import FlatlandEnv

###

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

###

N_ITERATIONS = 10
N_EPISODES_FOR_ITERATION = 100

###


def main():
    # raw_env = FlatlandEnv()
    # environment = tf_py_environment.TFPyEnvironment(raw_env)
    environment = FlatlandEnv()

    utils.validate_py_environment(environment, episodes=Configs.TRAIN_N_EPISODES)


###

if __name__ == '__main__':
    main()
