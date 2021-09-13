# from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen

from observators.tree import BinaryTreeObservator
from environments.keras import KerasEnvironment

import configs as Configs

###


def _prepare_observator():
    return BinaryTreeObservator(max_memory=Configs.OBS_TREE_N_NODES)


def _prepare_rail_generator():
    return sparse_rail_generator(
        seed=Configs.APP_SEED,
        max_num_cities=Configs.RAIL_ENV_N_CITIES,
        grid_mode=Configs.RAIL_ENV_CITIES_GRID_DISTRIBUTION,
        max_rails_between_cities=Configs.RAIL_ENV_MAX_RAILS_BETWEEN_CITIES,
        max_rails_in_city=Configs.RAIL_ENV_MAX_RAILS_IN_CITY
    )


def _prepare_schedule_generator():
    return sparse_schedule_generator(Configs.RAIL_ENV_SPEED_RATION_MAP)


def _prepare_malfunction_generator():
    stochastic_data = MalfunctionParameters(
        malfunction_rate=Configs.RAIL_ENV_MALFUNCTION_RATE,
        min_duration=Configs.RAIL_ENV_MALFUNCTION_MIN_DURATION,
        max_duration=Configs.RAIL_ENV_MALFUNCTION_MAX_DURATION
    )

    return ParamMalfunctionGen(stochastic_data)


###


def prepare_env():
    return KerasEnvironment(
        observator=_prepare_observator(),
        rail_generator=_prepare_rail_generator(),
        schedule_generator=_prepare_schedule_generator(),
        malfunction_generator=_prepare_malfunction_generator()
    )
