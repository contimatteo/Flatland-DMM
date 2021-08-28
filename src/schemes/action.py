from enum import IntEnum

from flatland.envs.rail_env import RailEnvActions

###

LowLevelAction = RailEnvActions


class HighLevelAction(IntEnum):
    STOP = 0
    LEFT_ORIENTED = 1
    RIGHT_ORIENTED = 2
