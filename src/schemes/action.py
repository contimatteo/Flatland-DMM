from typing import Tuple
from enum import IntEnum

import numpy as np

from flatland.envs.rail_env import RailEnvActions

###

LowLevelAction = RailEnvActions

###


class HighLevelAction(IntEnum):
    # changed order of values
    STOP = 1
    LEFT_ORIENTED = 0
    RIGHT_ORIENTED = 2

    def to_low_level_conv(self, orientation, possible_transitions):

        if self == self.STOP:
            return LowLevelAction.STOP_MOVING.value

        possible_actions = np.roll(possible_transitions, (1 - orientation) % 4)
        if possible_actions[self.value]:

            return LowLevelAction(self.value + 1).value
        else:
            return LowLevelAction.MOVE_FORWARD.value
