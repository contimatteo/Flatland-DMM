import numpy as np

from flatland.envs.rail_env import RailEnvActions

###

ACTIONS = np.unique([
    RailEnvActions.DO_NOTHING,  # implies change of direction in a dead-end!
    RailEnvActions.MOVE_LEFT,
    RailEnvActions.MOVE_FORWARD,
    RailEnvActions.MOVE_RIGHT,
    RailEnvActions.STOP_MOVING,
])

###


class BaseAgent:
    def __init__(self):
        self.available_actions = ACTIONS

    def initialize(self, *args, **kwargs):
        raise Exception('not implemented.')

    def act(self, *args, **kwargs):
        raise Exception('not implemented.')

    def step(self, *args, **kwargs):
        raise Exception('not implemented.')

    def save(self, *args, **kwargs):
        # TODO: store the current policy
        raise Exception('not implemented.')

    def load(self, *args, **kwargs):
        # TODO: load a policy
        raise Exception('not implemented.')
