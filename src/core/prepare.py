from typing import Tuple, List

from datetime import datetime
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.malfunction_generators import ParamMalfunctionGen
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_generators import RailGen
from flatland.envs.schedule_generators import ScheduleGenerator
from flatland.envs.schedule_generators import sparse_schedule_generator
from rl.callbacks import Callback
# from rl.callbacks import FileLogger
# from rl.callbacks import ModelIntervalCheckpoint
from rl.policy import Policy
from rl.policy import LinearAnnealedPolicy
from rl.policy import SoftmaxPolicy
from rl.policy import EpsGreedyQPolicy
from rl.policy import GreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

import configs as Configs

from core import MarlEnvironment
from marl.callbacks import FileLogger
from marl.callbacks import ModelIntervalCheckpoint
from networks import BaseNetwork
from networks import SequentialNetwork
from observators import BinaryTreeObservator

###


def _prepare_observator() -> ObservationBuilder:
    return BinaryTreeObservator(max_memory=Configs.OBS_TREE_N_NODES)


def _prepare_rail_generator() -> RailGen:
    return sparse_rail_generator(
        seed=Configs.APP_SEED,
        max_num_cities=Configs.RAIL_ENV_N_CITIES,
        grid_mode=Configs.RAIL_ENV_CITIES_GRID_DISTRIBUTION,
        max_rails_between_cities=Configs.RAIL_ENV_MAX_RAILS_BETWEEN_CITIES,
        max_rails_in_city=Configs.RAIL_ENV_MAX_RAILS_IN_CITY
    )


def _prepare_schedule_generator() -> ScheduleGenerator:
    return sparse_schedule_generator(Configs.RAIL_ENV_SPEED_RATION_MAP)


def _prepare_malfunction_generator() -> ParamMalfunctionGen:
    stochastic_data = MalfunctionParameters(
        malfunction_rate=Configs.RAIL_ENV_MALFUNCTION_RATE,
        min_duration=Configs.RAIL_ENV_MALFUNCTION_MIN_DURATION,
        max_duration=Configs.RAIL_ENV_MALFUNCTION_MAX_DURATION
    )

    return ParamMalfunctionGen(stochastic_data)


###


def prepare_env() -> MarlEnvironment:
    return MarlEnvironment(
        observator=_prepare_observator(),
        rail_generator=_prepare_rail_generator(),
        schedule_generator=_prepare_schedule_generator(),
        malfunction_generator=_prepare_malfunction_generator()
    )


def prepare_network(
    env: MarlEnvironment
) -> Tuple[BaseNetwork, optimizer_v2.OptimizerV2, List[str]]:
    network = SequentialNetwork(env.observation_space.shape, env.action_space.n)
    optimizer = Adam(learning_rate=Configs.DQN_AGENT_LEARNING_RATE)
    metrics = ['mae']

    return network, optimizer, metrics


def prepare_memory():
    return SequentialMemory(limit=Configs.DQN_AGENT_MEMORY_LIMIT, window_length=1)


def prepare_policy(policy_type: str = "eps-greedy", *args, **kwargs) -> Policy:
    policy = None

    if policy_type == "linear-annealed":
        policy = LinearAnnealedPolicy(*args, **kwargs)
    elif policy_type == "softmax":
        policy = SoftmaxPolicy(*args, **kwargs)
    elif policy_type == "eps-greedy":
        policy = EpsGreedyQPolicy(*args, **kwargs)
    elif policy_type == "greedy":
        policy = GreedyQPolicy(*args, **kwargs)
    elif policy_type == "boltzmann":
        policy = BoltzmannQPolicy(*args, **kwargs)
    elif policy_type == "max-boltzmann":
        policy = MaxBoltzmannQPolicy(*args, **kwargs)
    elif policy_type == "boltzmann-gumbel":
        policy = BoltzmannGumbelQPolicy(*args, **kwargs)

    if policy is None:
        raise Exception(f"invalid policy type '{policy_type}' value.")

    return policy


def prepare_callbacks(callback_types: List[str] = []) -> List[Callback]:
    callbacks = []

    # log_filename = 'tmp/dqn_log.json'
    # callbacks += [FileLogger(log_filename, interval=100)]

    interval = 1000
    interval_checkpoint_weights_filepath = './tmp/weights/sequential-1/intervals/{step}.h5f'
    callbacks += [ModelIntervalCheckpoint(interval_checkpoint_weights_filepath, interval=interval)]

    return callbacks
