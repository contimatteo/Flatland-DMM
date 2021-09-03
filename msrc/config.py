ENV_SEED = 1
ENV_WIDTH = 20
ENV_HEIGHT = 20

ENV_MAX_FRAMES = 100
ENV_MAX_EPISODES = 5

N_AGENTS = 2

RAIL_NR_START_GOAL = N_AGENTS + 2
RAIL_NR_EXTRA = 10
RAIL_MIN_DIST = 10
RAIL_MAX_DIST = 20

OBS_TREE_DEPTH = 1
OBS_MAX_VALUE = 1000

OBSERVED_NODE_PARAMS = [
    'dist_own_target_encountered',
    # 'dist_other_target_encountered',
    'dist_other_agent_encountered',
    'dist_potential_conflict',
    # 'dist_unusable_switch',
    'dist_to_next_branch',
    'dist_min_to_target',
    'num_agents_same_direction',
    'num_agents_opposite_direction',
    # 'num_agents_malfunctioning',
    # 'speed_min_fractional',
    # 'num_agents_ready_to_depart',
]
