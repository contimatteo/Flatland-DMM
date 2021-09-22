from typing import Tuple, List, Any

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_generators import MalfunctionParameters
from flatland.envs.malfunction_generators import ParamMalfunctionGen
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_generators import RailGen
from flatland.envs.schedule_generators import ScheduleGenerator
from flatland.envs.schedule_generators import sparse_schedule_generator
from rl.callbacks import Callback
from rl.policy import Policy
from rl.policy import LinearAnnealedPolicy
from rl.policy import SoftmaxPolicy
from rl.policy import EpsGreedyQPolicy
from rl.policy import GreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
# from wandb.keras import WandbCallback

from configs import configurator as Configs

from core import MarlEnvironment
from core import BinaryTreeObservator
# from marl.callbacks import FileLogger
# from marl.callbacks import ModelIntervalCheckpoint
from marl.callbacks import WandbLogger
from networks import BaseNetwork
from networks import SequentialNetwork1

###


def _prepare_observator() -> ObservationBuilder:
    return BinaryTreeObservator(max_memory=Configs.OBS_TREE_N_NODES)


def _prepare_rail_generator() -> RailGen:
    return sparse_rail_generator(
        seed=Configs.SEED,
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


def prepare_network(env: MarlEnvironment) -> BaseNetwork:
    network = None

    ctype = Configs.NN_TYPE
    params = Configs.NN_PARAMS

    if ctype == "sequential-1":
        network = SequentialNetwork1(env.observation_space.shape, env.action_space.n, **params)

    if network is None:
        raise Exception(f"invalid network type '{ctype}' value.")

    return network


def prepare_optimizer() -> optimizer_v2.OptimizerV2:
    optimizer = None

    ctype = Configs.NN_OPTIMIZER_TYPE
    params = Configs.NN_OPTIMIZER_PARAMS

    if ctype == "adam":
        optimizer = Adam(**params)
    elif ctype == 'sgd':
        optimizer = SGD(**params)

    if optimizer is None:
        raise Exception(f"invalid optimizer type '{ctype}' value.")

    return optimizer


def prepare_metrics() -> List[str]:
    metrics = Configs.NN_METRICS

    if 'mae' not in metrics:
        metrics += ['mae']

    return metrics


def prepare_memory():
    return SequentialMemory(limit=Configs.AGENT_MEMORY_LIMIT, window_length=1)


def prepare_policy() -> Policy:
    policy = None

    ctype = Configs.POLICY_TYPE
    params = Configs.POLICY_PARAMS

    if ctype == "linear-annealed":
        policy = LinearAnnealedPolicy(**params)
    elif ctype == "softmax":
        policy = SoftmaxPolicy()
    elif ctype == "eps-greedy":
        policy = EpsGreedyQPolicy(**params)
    elif ctype == "greedy":
        policy = GreedyQPolicy(**params)
    elif ctype == "boltzmann":
        policy = BoltzmannQPolicy(**params)
    elif ctype == "max-boltzmann":
        policy = MaxBoltzmannQPolicy(**params)
    elif ctype == "boltzmann-gumbel":
        policy = BoltzmannGumbelQPolicy(**params)

    if policy is None:
        raise Exception(f"invalid policy type '{ctype}' value.")

    return policy


def prepare_callbacks(types: List[str], network: BaseNetwork) -> List[Callback]:
    callbacks = []

    # callbacks += [WandbCallback()]
    callbacks += [WandbLogger(project='flatland', group=Configs.CONFIG_UUID, entity='flatland-dmm')]

    ### TODO: [@matteo]

    # log_filename = './tmp/logs/dqn/'
    # Path(log_filename).mkdir(parents=True, exist_ok=True)
    # log_filename += '{}.json'.format(datetime.now()) # .strftime("%s"))
    # Path(log_filename).touch(exist_ok=True)
    # callbacks += [FileLogger(log_filename, interval=100)]

    # interval=int(Configs.TRAIN_N_MAX_STEPS_FOR_EPISODE)
    # weights_intervals_filepath = network.weights_intervals_file_url
    # callbacks += [ModelIntervalCheckpoint(weights_intervals_filepath, interval=interval)]

    return callbacks
