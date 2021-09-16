from typing import Tuple
from enum import IntEnum

import numpy as np

from flatland.envs.rail_env import RailEnvActions

###

LowLevelAction = RailEnvActions

###


class HighLevelAction(IntEnum):

    LEFT_ORIENTED = 0
    STOP = 1
    RIGHT_ORIENTED = 2

    #

    def to_low_level(self, orientation: int, possible_transitions: Tuple[bool]) -> LowLevelAction:
        if self == self.STOP:
            return LowLevelAction.STOP_MOVING

        possible_actions = np.roll(possible_transitions, (1 - orientation) % 4)

        if possible_actions[self.value] is True:
            return LowLevelAction(self.value + 1)
        else:
            return LowLevelAction.MOVE_FORWARD
