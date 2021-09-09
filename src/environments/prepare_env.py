from observators.tree import BinaryTreeObservator
from environments.keras_env import KerasEnvironment

from typing import Dict, Any, Tuple, List
from flatland.envs.rail_env import RailEnv, EnvAgent, Grid4TransitionsEnum
from flatland.envs.rail_generators import random_rail_generator, sparse_rail_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen

import configs as Configs

def prepare_env():
    obs = BinaryTreeObservator(max_memory = Configs.N_OBS_NODE)


    rail_generator = sparse_rail_generator(
            max_num_cities = Configs.N_CITIES,
            seed = Configs.SEED,
            grid_mode = Configs.GRID_DISTRIBUTION_OF_CITIES,
            max_rails_between_cities = Configs.MAX_RAIL_BETWEEN_CITIES,
            max_rails_in_city = Configs.MAX_RAIL_IN_CITY
        )


    schedule_generator = sparse_schedule_generator(Configs.SPEED_RATION_MAP)


    stochastic_data = MalfunctionParameters(
        malfunction_rate=Configs.MALFUNCTION_RATE,
        min_duration=Configs.MALFUNCTION_MIN_DURATION,
        max_duration=Configs.MALFUNCTION_MAX_DURATION
    )
    malfunction_generator = ParamMalfunctionGen(stochastic_data)


    env = KerasEnvironment(
        observator=obs,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        malfunction_generator=malfunction_generator
    )

    return env