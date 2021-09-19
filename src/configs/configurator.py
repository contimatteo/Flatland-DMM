from typing import Mapping
from pathlib import Path
import warnings

import json

from utils.obs_node import Node

###################################################################################################
###################################################################################################

#########
## APP ##
#########

APP_DEBUG: bool = None
APP_SEED: int = 100
N_AGENTS: int = None
N_ACTIONS: int = 3


RAIL_ENV_MAP_WIDTH: int = None
RAIL_ENV_MAP_HEIGHT: int = None
RAIL_ENV_N_CITIES: int = None
RAIL_ENV_MAX_RAILS_IN_CITY: int = None
RAIL_ENV_MAX_RAILS_BETWEEN_CITIES: int = None
RAIL_ENV_CITIES_GRID_DISTRIBUTION: bool = None
RAIL_ENV_SPEED_RATION_MAP: Mapping[float, float] = {
    1.: 0.25,  # Fast passenger train
    1. / 2.: 0.25,  # Fast freight train
    1. / 3.: 0.25,  # Slow commuter train
    1. / 4.: 0.25,  # Slow freight train
}
RAIL_ENV_MALFUNCTION_RATE: float = None
RAIL_ENV_MALFUNCTION_MIN_DURATION: int = None
RAIL_ENV_MALFUNCTION_MAX_DURATION: int = None
RAIL_ENV_REMOVE_AGENTS_AT_TARGET: bool = True

EMULATOR_ACTIVE: bool = None
EMULATOR_WINDOW_WIDTH: int = None
EMULATOR_WINDOW_HEIGHT: int = None
EMULATOR_STEP_TIMEBREAK_SECONDS: int = None

OBS_TREE_N_NODES: int = None
OBS_TREE_NODE_N_FEATURES: int = Node.get_n_of_features() # TODO: change this
OBS_LENGTH: int = None

POLICY_TYPE: str = None
POLICY_PARAMETERS: dict = None

DQN_AGENT_TEST_VERBOSE: int = None
DQN_AGENT_TRAIN_VERBOSE: int = None
DQN_AGENT_MEMORY_LIMIT: int = None
DQN_AGENT_LEARNING_RATE: float = None
DQN_AGENT_TARGET_MODEL_UPDATE: float = None

TRAIN_N_MIN_ATTEMPTS: int = None
TRAIN_LOG_INTERVAL: int = None
TRAIN_N_STEPS_WARMUP: int = None
TRAIN_N_MAX_STEPS_FOR_EPISODE: int = None
TRAIN_N_STEPS: int = None

TEST_N_ATTEMPTS: int = None
TEST_N_MAX_STEPS_FOR_EPISODE: int = None

###################################################################################################
###################################################################################################


def get_configs_from_file(filepath):
    if Path(filepath).is_file() is False:
        warnings.warn("Configurations json file not found. Using default configs.")
        return

    file_stream = open(filepath, mode="r", encoding="utf-8")
    configurations = json.load(file_stream)
    file_stream.close()

    if configurations is None or not isinstance(configurations, list):
        warnings.warn("Configurations json malformed. See the `run.train.json` example file.")
        configurations = []

    return configurations


def reset():
    global APP_DEBUG, N_AGENTS
    global RAIL_ENV_MAP_WIDTH, RAIL_ENV_MAP_HEIGHT, RAIL_ENV_N_CITIES, RAIL_ENV_MAX_RAILS_IN_CITY
    global RAIL_ENV_MAX_RAILS_BETWEEN_CITIES, RAIL_ENV_CITIES_GRID_DISTRIBUTION
    global RAIL_ENV_MALFUNCTION_RATE, RAIL_ENV_MALFUNCTION_MIN_DURATION
    global RAIL_ENV_MALFUNCTION_MAX_DURATION
    global EMULATOR_ACTIVE, EMULATOR_WINDOW_WIDTH, EMULATOR_WINDOW_HEIGHT, EMULATOR_STEP_TIMEBREAK_SECONDS
    global OBS_TREE_N_NODES, OBS_LENGTH
    global POLICY_TYPE, POLICY_PARAMETERS
    global DQN_AGENT_TEST_VERBOSE, DQN_AGENT_TRAIN_VERBOSE, DQN_AGENT_MEMORY_LIMIT
    global DQN_AGENT_LEARNING_RATE, DQN_AGENT_TARGET_MODEL_UPDATE
    global TRAIN_N_MIN_ATTEMPTS, TRAIN_LOG_INTERVAL, TRAIN_N_STEPS_WARMUP
    global TRAIN_N_MAX_STEPS_FOR_EPISODE, TRAIN_N_STEPS
    global TEST_N_ATTEMPTS, TEST_N_MAX_STEPS_FOR_EPISODE

    ###

    APP_DEBUG = True
    N_AGENTS = 2

    RAIL_ENV_MAP_WIDTH = 4 * 7
    RAIL_ENV_MAP_HEIGHT = 3 * 7
    RAIL_ENV_N_CITIES = 2
    RAIL_ENV_MAX_RAILS_IN_CITY = 1
    RAIL_ENV_MAX_RAILS_BETWEEN_CITIES = 1
    RAIL_ENV_CITIES_GRID_DISTRIBUTION = False
    RAIL_ENV_MALFUNCTION_RATE = 1 / 10000
    RAIL_ENV_MALFUNCTION_MIN_DURATION = 15
    RAIL_ENV_MALFUNCTION_MAX_DURATION = 50

    EMULATOR_ACTIVE = False
    EMULATOR_WINDOW_WIDTH = 1200
    EMULATOR_WINDOW_HEIGHT = 1200
    EMULATOR_STEP_TIMEBREAK_SECONDS = 0

    OBS_TREE_N_NODES = 1 + 2
    OBS_LENGTH = int(OBS_TREE_NODE_N_FEATURES * OBS_TREE_N_NODES + 1)

    POLICY_TYPE = 'eps-greedy'
    POLICY_PARAMETERS = {'eps': 0.05}

    DQN_AGENT_TEST_VERBOSE = 1
    DQN_AGENT_TRAIN_VERBOSE = 1
    DQN_AGENT_MEMORY_LIMIT = 10000
    DQN_AGENT_LEARNING_RATE = 1e-3
    DQN_AGENT_TARGET_MODEL_UPDATE = 100

    TRAIN_N_MIN_ATTEMPTS = 1
    TRAIN_LOG_INTERVAL = 250
    TRAIN_N_STEPS_WARMUP = 100
    TRAIN_N_MAX_STEPS_FOR_EPISODE = 500
    TRAIN_N_STEPS = int(TRAIN_N_MIN_ATTEMPTS * TRAIN_N_MAX_STEPS_FOR_EPISODE)

    TEST_N_ATTEMPTS = 5
    TEST_N_MAX_STEPS_FOR_EPISODE = 1500


def load_configs(configurations):
    global APP_DEBUG, N_AGENTS
    global RAIL_ENV_MAP_WIDTH, RAIL_ENV_MAP_HEIGHT, RAIL_ENV_N_CITIES, RAIL_ENV_MAX_RAILS_IN_CITY
    global RAIL_ENV_MAX_RAILS_BETWEEN_CITIES, RAIL_ENV_CITIES_GRID_DISTRIBUTION
    global RAIL_ENV_MALFUNCTION_RATE, RAIL_ENV_MALFUNCTION_MIN_DURATION
    global RAIL_ENV_MALFUNCTION_MAX_DURATION
    global EMULATOR_ACTIVE, EMULATOR_WINDOW_WIDTH, EMULATOR_WINDOW_HEIGHT, EMULATOR_STEP_TIMEBREAK_SECONDS
    global OBS_TREE_N_NODES
    global POLICY_TYPE, POLICY_PARAMETERS
    global DQN_AGENT_TEST_VERBOSE, DQN_AGENT_TRAIN_VERBOSE, DQN_AGENT_MEMORY_LIMIT
    global DQN_AGENT_LEARNING_RATE, DQN_AGENT_TARGET_MODEL_UPDATE
    global TRAIN_N_MIN_ATTEMPTS, TRAIN_LOG_INTERVAL, TRAIN_N_STEPS_WARMUP
    global TRAIN_N_MAX_STEPS_FOR_EPISODE
    global TEST_N_ATTEMPTS, TEST_N_MAX_STEPS_FOR_EPISODE

    ###

    if 'policy' in configurations:
        POLICY_TYPE = configurations['policy']['type']
        POLICY_PARAMETERS = configurations['policy']['parameters']
