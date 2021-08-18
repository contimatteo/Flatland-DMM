#

DEBUG: bool = True
RANDOM_SEED: int = 100

##################
## Flatland Env ##
##################

N_OF_AGENTS: int = 1

RAIL_ENV_WIDTH: int = 10
RAIL_ENV_HEIGHT: int = 10
RAIL_ENV_N_CELLS: int = RAIL_ENV_WIDTH * RAIL_ENV_HEIGHT

TRAIN_N_EPISODES: int = 4
# TRAIN_N_MAX_ATTEMPTS_FOR_EPISODE: int = 100

##############
## Emulator ##
##############

EMULATOR_ACTIVE: bool = True
EMULATOR_WINDOW_WIDTH: int = 600
EMULATOR_WINDOW_HEIGHT: int = 600
EMULATOR_STEP_TIMEBREAK_SECONDS: int = 0.2

##################
## Observations ##
##################

OBS_TREE_NODE_N_FEATURES: int = 12
OBS_TREE_N_NODES: int = 1 + 2
OBS_TREE_N_FEATURES: int = OBS_TREE_N_NODES * OBS_TREE_NODE_N_FEATURES
