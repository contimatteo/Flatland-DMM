from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
import config
from msrc.observer import TreeTensorObserver


class FLEnvironment:
    def __init__(self):
        # RAIL GENERATOR
        self._rail_gen = complex_rail_generator(
            nr_start_goal=config.RAIL_NR_START_GOAL,
            nr_extra=config.RAIL_NR_EXTRA,
            min_dist=config.RAIL_MIN_DIST,
            max_dist=config.RAIL_MAX_DIST,
            seed=config.ENV_SEED
        )

        # OBSERVATION
        self._obs_builder = TreeTensorObserver()

        # ENVIRONMENT
        self._env = RailEnv(
            width=config.ENV_WIDTH,
            height=config.ENV_HEIGHT,
            number_of_agents=config.N_AGENTS,
            remove_agents_at_target=True,
            obs_builder_object=self._obs_builder,
            random_seed=config.ENV_SEED,
            rail_generator=self._rail_gen
        )

    def get_env(self):
        return self._env

