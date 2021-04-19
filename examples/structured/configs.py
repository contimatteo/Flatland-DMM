#

DEBUG = False
RANDOM_SEED = 100

#
# EMULATOR
#

EMULATOR_ACTIVE = False
EMULATOR_WINDOW_WIDTH = 1200
EMULATOR_WINDOW_HEIGHT = 1200

#
# RAIL ENVIRONMENT
#

RAIL_ENV_WIDTH = 20
RAIL_ENV_HEIGHT = 20
RAIL_ENV_N_CELLS = RAIL_ENV_WIDTH * RAIL_ENV_HEIGHT

NUMBER_OF_AGENTS = 1

TRAIN_N_ATTEMPTS = 1
TRAIN_N_MAX_EPISODES = 1000

#
# OBSERVATORS
#

OBSERVATOR_TREE_N_NODES = 1 + 1
OBSERVATION_TREE_N_ATTRIBUTES = 13
OBSERVATION_TREE_STATE_SIZE = OBSERVATOR_TREE_N_NODES * OBSERVATION_TREE_N_ATTRIBUTES
