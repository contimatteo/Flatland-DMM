
DEBUG = True
SEED = 100

#######################
## Flatland Rail Gen ##
#######################

N_CITIES = 2
GRID_DISTRIBUTION_OF_CITIES = False
MAX_RAIL_BETWEEN_CITIES = 1
MAX_RAIL_IN_CITY = 1

################################
## Flatland Rail Schedule Gen ##
################################

SPEED_RATION_MAP = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

##############################
## Flatland Malfunction Par ##
##############################

MALFUNCTION_RATE = 1/10000
MALFUNCTION_MIN_DURATION = 15
MALFUNCTION_MAX_DURATION = 50

###########################
## Observator Parameters ##
###########################

N_OBS_NODE = 10

#######################
## Flatland Rail Env ##
#######################

MAP_WIDTH = 4 * 7
MAP_HEIGHT = 3 * 7
N_AGENTS = 1
REMOVE_AGENTS_AT_TARGET = True

###########################
## Flatland Env Renderer ##
###########################

EMULATOR_ACTIVE: bool = False
EMULATOR_WINDOW_WIDTH: int = 1000
EMULATOR_WINDOW_HEIGHT: int = 1000
EMULATOR_STEP_TIMEBREAK_SECONDS: int = 0.3

###########################
## Agent Hyperparameters ##
###########################

N_ITERATIONS = 20000

INITIAL_COLLECT_STEPS = 100
COLLECT_STEPS_PER_ITERATION = 1
REPLAY_BUFFER_MAX_LENGTH = 100000

BATCH_SIZE = 64
LOG_INTERVAL = 200

N_EVAL_EPISODES = 10
EVAL_INTERVAL = 1000

########################
## Network Parameters ##
########################

ACTIVATION_HIDDEN_L = 'relu'
ACTIVATION_OUTPUT_L = 'relu'
LOSS_FUNCTION = 'mean_squared_error'
LEARNING_RATE = 1e-3
ACTION_SIZE = 3

#########################
## Training Parameters ##
#########################

TRAIN_N_ATTEMPTS = 1
TRAIN_N_EPISODES = 100

RANDOM_SEED = 0