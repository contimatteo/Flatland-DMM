from typing import Dict

import numpy as np

from rl.core import Env
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from schemes.action import HighLevelAction
from schemes.node import Node
from utils.environment import RailEnvWrapper

###


class KerasEnvironment(Env):
    def __init__(self, observator, rail_generator, schedule_generator, malfunction_generator):
        super().__init__()

        self.observator = observator

        self._env = RailEnvWrapper(
            observator=self.observator,
            rail_generator=rail_generator,
            schedule_generator=schedule_generator,
            malfunction_generator=malfunction_generator
        )

    #

    def observation_spec(self):
        return array_spec.ArraySpec(
            shape=(Node.get_n_of_features(), ),
            dtype=np.int32,
            name='observation',
        )

    def action_spec(self):
        raw_actions_values = list(map(int, HighLevelAction))

        return array_spec.BoundedArraySpec(
            name='action',
            dtype=np.int32,
            shape=(self._env.n_agents, ),
            minimum=min(raw_actions_values),
            maximum=max(raw_actions_values),
        )

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())

    #

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def render(self, mode='human', close=False):
        pass

    def reset(self):
        return self._env.reset()

    def step(self, action: Dict[int, HighLevelAction]):
        return self._env.step(action)
