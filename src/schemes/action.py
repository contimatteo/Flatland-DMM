from enum import IntEnum

from flatland.envs.rail_env import RailEnvActions

###

LowLevelAction = RailEnvActions

###


class HighLevelAction(IntEnum):
    STOP = 0
    LEFT_ORIENTED_ACTION = 1
    RIGHT_ORIENTED_ACTION = 2
