IS_TRAINING = True
PRELOAD_MODEL = False

N_TRAINS = 4

ENV_SEED = 112
ENV_WIDTH = 10
ENV_HEIGHT = 10

ENV_MAX_TIMESTEPS = 100
ENV_REGENERATION_FREQUENCY = 12

TRAINING_EPISODES = 5000
EVALUATION_EPISODES = 10

RAIL_NR_START_GOAL = N_TRAINS + 2
RAIL_NR_EXTRA = 10
RAIL_MIN_DIST = 10
RAIL_MAX_DIST = 20

OBS_TREE_DEPTH = 3
OBS_MAX_VALUE = 100

OBSERVED_NODE_PARAMS = [
    'dist_own_target_encountered',
    # 'dist_other_target_encountered',
    'dist_other_agent_encountered',
    'dist_potential_conflict',
    'dist_unusable_switch',
    'dist_to_next_branch',
    'dist_min_to_target',
    'num_agents_same_direction',
    'num_agents_opposite_direction',
    # 'num_agents_malfunctioning',
    # 'speed_min_fractional',
    # 'num_agents_ready_to_depart',
]
