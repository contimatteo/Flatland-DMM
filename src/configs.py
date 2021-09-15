from typing import Mapping

#########
## APP ##
#########

APP_DEBUG: bool = True
APP_SEED: int = 100

##########
## BASE ##
##########

N_AGENTS = 2
N_ACTIONS = 3

TRAIN_N_STEPS = 3000
TRAIN_N_MAX_EPISODE_STEPS = 9000

##############
## RAIL ENV ##
##############

RAIL_ENV_MAP_WIDTH: int = 4 * 7
RAIL_ENV_MAP_HEIGHT: int = 3 * 7

RAIL_ENV_N_CITIES: int = 2
RAIL_ENV_MAX_RAILS_IN_CITY: int = 1
RAIL_ENV_MAX_RAILS_BETWEEN_CITIES: int = 1
RAIL_ENV_CITIES_GRID_DISTRIBUTION: bool = False

RAIL_ENV_SPEED_RATION_MAP: Mapping[float, float] = {
    1.: 0.25,  # Fast passenger train
    1. / 2.: 0.25,  # Fast freight train
    1. / 3.: 0.25,  # Slow commuter train
    1. / 4.: 0.25,  # Slow freight train
}

RAIL_ENV_MALFUNCTION_RATE: float = 1 / 10000
RAIL_ENV_MALFUNCTION_MIN_DURATION: int = 15
RAIL_ENV_MALFUNCTION_MAX_DURATION: int = 50

RAIL_ENV_REMOVE_AGENTS_AT_TARGET: bool = True

##############
## EMULATOR ##
##############

EMULATOR_ACTIVE: bool = False
EMULATOR_WINDOW_WIDTH: int = 1200
EMULATOR_WINDOW_HEIGHT: int = 1200
EMULATOR_STEP_TIMEBREAK_SECONDS: int = 0.3

#########
## OBS ##
#########

OBS_TREE_N_NODES: int = 1 + 2
OBS_TREE_NODE_N_FEATURES: int = 13

###############
## DQN-AGENT ##
###############

DQN_AGENT_VERBOSE: int = 1
DQN_AGENT_MEMORY_LIMIT: int = 25000
DQN_AGENT_LEARNING_RATE: float = 1e-3
DQN_AGENT_TARGET_MODEL_UPDATE: float = 1e-2
