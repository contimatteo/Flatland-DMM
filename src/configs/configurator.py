from typing import Mapping
from pathlib import Path
import warnings

import json

from utils.obs_node import Node

###################################################################################################
###################################################################################################

N_ACTIONS: int = 3

RAIL_ENV_SPEED_RATION_MAP: Mapping[float, float] = {
    1.: 0.25,  # Fast passenger train
    1. / 2.: 0.25,  # Fast freight train
    1. / 3.: 0.25,  # Slow commuter train
    1. / 4.: 0.25,  # Slow freight train
}
RAIL_ENV_REMOVE_AGENTS_AT_TARGET: bool = True

###

SEED: int = None
DEBUG: bool = None
N_AGENTS: int = None
CONFIG_UUID: str = None

RAIL_ENV_MAP_WIDTH: int = None
RAIL_ENV_MAP_HEIGHT: int = None
RAIL_ENV_N_CITIES: int = None
RAIL_ENV_MAX_RAILS_IN_CITY: int = None
RAIL_ENV_MAX_RAILS_BETWEEN_CITIES: int = None
RAIL_ENV_CITIES_GRID_DISTRIBUTION: bool = None
RAIL_ENV_MALFUNCTION_RATE: float = None
RAIL_ENV_MALFUNCTION_MIN_DURATION: int = None
RAIL_ENV_MALFUNCTION_MAX_DURATION: int = None

EMULATOR_ACTIVE: bool = None
EMULATOR_WINDOW_WIDTH: int = None
EMULATOR_WINDOW_HEIGHT: int = None
EMULATOR_STEP_TIMEBREAK_SECONDS: int = None

OBS_TREE_N_NODES: int = None

POLICY_TYPE: str = None
POLICY_PARAMS: dict = None

AGENT_TYPE: str = None
AGENT_MEMORY_LIMIT: int = None
AGENT_PARAMS: dict = None

NN_TYPE: str = None
NN_PARAMS: dict = None
NN_METRICS: list = None
NN_OPTIMIZER_TYPE: str = None
NN_OPTIMIZER_PARAMS: dict = None

TRAIN_VERBOSE: int = None
TRAIN_N_MIN_ATTEMPTS: int = None
TRAIN_LOG_INTERVAL: int = None
TRAIN_N_MAX_STEPS_FOR_EPISODE: int = None
TRAIN_N_STEPS: int = None
TRAIN_CALLBACKS: list = None

TEST_VERBOSE: int = None
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
    global SEED, DEBUG, N_AGENTS, CONFIG_UUID
    global RAIL_ENV_MAP_WIDTH, RAIL_ENV_MAP_HEIGHT, RAIL_ENV_N_CITIES, RAIL_ENV_MAX_RAILS_IN_CITY
    global RAIL_ENV_MAX_RAILS_BETWEEN_CITIES, RAIL_ENV_CITIES_GRID_DISTRIBUTION
    global RAIL_ENV_MALFUNCTION_RATE, RAIL_ENV_MALFUNCTION_MIN_DURATION, RAIL_ENV_MALFUNCTION_MAX_DURATION
    global EMULATOR_ACTIVE, EMULATOR_WINDOW_WIDTH, EMULATOR_WINDOW_HEIGHT, EMULATOR_STEP_TIMEBREAK_SECONDS
    global OBS_TREE_N_NODES
    global POLICY_TYPE, POLICY_PARAMS
    global AGENT_TYPE, AGENT_MEMORY_LIMIT, AGENT_PARAMS
    global NN_TYPE, NN_PARAMS, NN_METRICS, NN_OPTIMIZER_TYPE, NN_OPTIMIZER_PARAMS
    global TRAIN_VERBOSE, TRAIN_N_MIN_ATTEMPTS, TRAIN_LOG_INTERVAL
    global TRAIN_N_MAX_STEPS_FOR_EPISODE, TRAIN_N_STEPS, TRAIN_CALLBACKS
    global TEST_VERBOSE, TEST_N_ATTEMPTS, TEST_N_MAX_STEPS_FOR_EPISODE

    ###

    SEED = 1
    DEBUG = False
    N_AGENTS = 2
    CONFIG_UUID = 'default'

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

    POLICY_TYPE = 'eps-greedy'
    POLICY_PARAMS = {'eps': 0.05}

    AGENT_TYPE = 'dqn'
    AGENT_MEMORY_LIMIT = 10000
    AGENT_PARAMS = {
        "dueling_type": 'avg',
        "nb_steps_warmup": 100,
        "target_model_update": 50,
        "enable_double_dqn": False,
        "enable_dueling_network": False,
    }

    NN_TYPE = 'sequential-1'
    NN_PARAMS = {}
    NN_METRICS = ["mae"]
    NN_OPTIMIZER_TYPE = 'adam'
    NN_OPTIMIZER_PARAMS = {"learning_rate": 0.001}

    TRAIN_VERBOSE = 1
    TRAIN_N_MIN_ATTEMPTS = 5
    TRAIN_LOG_INTERVAL = 500
    TRAIN_N_MAX_STEPS_FOR_EPISODE = 2000
    TRAIN_CALLBACKS = []
    TRAIN_N_STEPS = int(TRAIN_N_MIN_ATTEMPTS * TRAIN_N_MAX_STEPS_FOR_EPISODE)

    TEST_VERBOSE = 1
    TEST_N_ATTEMPTS = 5
    TEST_N_MAX_STEPS_FOR_EPISODE = 1500


def load_configs(configurations):
    global SEED, DEBUG, N_AGENTS, CONFIG_UUID
    global RAIL_ENV_MAP_WIDTH, RAIL_ENV_MAP_HEIGHT, RAIL_ENV_N_CITIES, RAIL_ENV_MAX_RAILS_IN_CITY
    global RAIL_ENV_MAX_RAILS_BETWEEN_CITIES, RAIL_ENV_CITIES_GRID_DISTRIBUTION
    global RAIL_ENV_MALFUNCTION_RATE, RAIL_ENV_MALFUNCTION_MIN_DURATION, RAIL_ENV_MALFUNCTION_MAX_DURATION
    global EMULATOR_ACTIVE, EMULATOR_WINDOW_WIDTH, EMULATOR_WINDOW_HEIGHT, EMULATOR_STEP_TIMEBREAK_SECONDS
    global OBS_TREE_N_NODES
    global POLICY_TYPE, POLICY_PARAMS
    global AGENT_TYPE, AGENT_MEMORY_LIMIT, AGENT_PARAMS
    global NN_TYPE, NN_PARAMS, NN_METRICS, NN_OPTIMIZER_TYPE, NN_OPTIMIZER_PARAMS
    global TRAIN_VERBOSE, TRAIN_N_MIN_ATTEMPTS, TRAIN_LOG_INTERVAL
    global TRAIN_N_MAX_STEPS_FOR_EPISODE, TRAIN_N_STEPS, TRAIN_CALLBACKS
    global TEST_VERBOSE, TEST_N_ATTEMPTS, TEST_N_MAX_STEPS_FOR_EPISODE

    ###

    SEED = configurations['seed']
    DEBUG = bool(configurations['seed'])
    N_AGENTS = configurations['n_agents']
    CONFIG_UUID = configurations['config_uuid']

    assert isinstance(SEED, int) and SEED > 0
    assert isinstance(DEBUG, bool)
    assert isinstance(N_AGENTS, int) and N_AGENTS > 0
    assert isinstance(CONFIG_UUID, str)

    if 'rail-env' in configurations:
        RAIL_ENV_MAP_WIDTH = configurations['rail-env']['map_width']
        RAIL_ENV_MAP_HEIGHT = configurations['rail-env']['map_height']
        RAIL_ENV_N_CITIES = configurations['rail-env']['n_cities']
        RAIL_ENV_MAX_RAILS_IN_CITY = configurations['rail-env']['max_rails_in_city']
        RAIL_ENV_MAX_RAILS_BETWEEN_CITIES = configurations['rail-env']['max_rails_between_cities']
        RAIL_ENV_CITIES_GRID_DISTRIBUTION = configurations['rail-env']['cities_grid_distribution']
        RAIL_ENV_MALFUNCTION_RATE = configurations['rail-env']['malfunction_rate']
        RAIL_ENV_MALFUNCTION_MIN_DURATION = configurations['rail-env']['malfunction_min_duration']
        RAIL_ENV_MALFUNCTION_MAX_DURATION = configurations['rail-env']['malfunction_max_duration']

    if 'emulator' in configurations:
        EMULATOR_ACTIVE = configurations['emulator']['active']
        EMULATOR_WINDOW_WIDTH = configurations['emulator']['window_width']
        EMULATOR_WINDOW_HEIGHT = configurations['emulator']['window_height']
        EMULATOR_STEP_TIMEBREAK_SECONDS = configurations['emulator']['step_timebreak_seconds']

    if 'observator' in configurations:
        OBS_TREE_N_NODES = configurations['observator']['n_nodes']

    if 'policy' in configurations:
        POLICY_TYPE = configurations['policy']['type']
        POLICY_PARAMS = configurations['policy']['parameters']

    if 'agent' in configurations:
        AGENT_TYPE = configurations['agent']['type']
        AGENT_MEMORY_LIMIT = configurations['agent']['memory_limit']
        AGENT_PARAMS = configurations['agent']['parameters']

    if 'network' in configurations:
        NN_TYPE = configurations['network']['type']
        NN_PARAMS = configurations['network']['parameters']
        NN_METRICS = configurations['network']['metrics']
        if 'optimizer' in configurations['network']:
            NN_OPTIMIZER_TYPE = configurations['network']['optimizer']['type']
            NN_OPTIMIZER_PARAMS = configurations['network']['optimizer']['parameters']

    if 'train' in configurations:
        TRAIN_VERBOSE = configurations['train']['verbose']
        TRAIN_N_MIN_ATTEMPTS = configurations['train']['n_min_attempts']
        TRAIN_LOG_INTERVAL = configurations['train']['log_interval']
        TRAIN_N_MAX_STEPS_FOR_EPISODE = configurations['train']['nb_max_episode_steps']
        TRAIN_CALLBACKS = configurations['train']['callbacks']
        TRAIN_N_STEPS = int(TRAIN_N_MIN_ATTEMPTS * TRAIN_N_MAX_STEPS_FOR_EPISODE)
