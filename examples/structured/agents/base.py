import numpy as np

from flatland.envs.rail_env import RailEnvActions

###

ACTIONS = set([
    RailEnvActions.DO_NOTHING,
    RailEnvActions.MOVE_LEFT,
    RailEnvActions.MOVE_FORWARD,
    RailEnvActions.MOVE_RIGHT,
    RailEnvActions.STOP_MOVING,
])

###


class BaseAgent:
    def __init__(self):
        self.weights_filename = 'weights'
        self.available_actions = ACTIONS

    def initialize(self, *params):
        raise Exception('not implemented.')

    def act(self, *params):
        raise Exception('not implemented.')

    def step(self, *params):
        raise Exception('not implemented.')

    def save(self, *params):
        # TODO: store the current policy
        # self.weights_filename ...
        raise Exception('not implemented.')

    def load(self, *params):
        # TODO: load a policy
        # self.weights_filename ...
        raise Exception('not implemented.')
