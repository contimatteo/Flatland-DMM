from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

import configs as Configs

from environments.pyenv import FlatlandEnvironmentSingleAgent

###

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

###

###


def main():
    # raw_env = FlatlandEnvironmentSingleAgent()
    # environment = tf_py_environment.TFPyEnvironment(raw_env)
    environment = FlatlandEnvironmentSingleAgent()

    utils.validate_py_environment(environment, episodes=Configs.TRAIN_N_EPISODES)


###

if __name__ == '__main__':
    main()
