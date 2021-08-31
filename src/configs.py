#

DEBUG: bool = True
RANDOM_SEED: int = 100

#######################
## Flatland Rail Env ##
#######################

RAIL_ENV_WIDTH: int = 10
RAIL_ENV_HEIGHT: int = 10
RAIL_ENV_N_CELLS: int = RAIL_ENV_WIDTH * RAIL_ENV_HEIGHT

##############
## Emulator ##
##############

EMULATOR_ACTIVE: bool = False
EMULATOR_WINDOW_WIDTH: int = 1000
EMULATOR_WINDOW_HEIGHT: int = 1000
EMULATOR_STEP_TIMEBREAK_SECONDS: int = 0.3

##################
## Observations ##
##################

OBS_TREE_NODE_N_FEATURES: int = 14
OBS_TREE_N_NODES: int = 1 + 2
OBS_TREE_N_FEATURES: int = (OBS_TREE_N_NODES - 1) * OBS_TREE_NODE_N_FEATURES

##############
## Training ##
##############

N_OF_AGENTS: int = 2

TRAIN_N_ATTEMPTS = 1
TRAIN_N_EPISODES = 100
