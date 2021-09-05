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

    def to_low_level_conv(self, orientation, possible_transitions):

        if self == self.STOP:
            return LowLevelAction.STOP_MOVING.value

        possible_actions = np.roll(possible_transitions, (1 - orientation) % 4)
        if possible_actions[self.value]:

            return LowLevelAction(self.value + 1).value
        else:
            return LowLevelAction.MOVE_FORWARD.value

    #

    def to_low_level(self, orientation: int, possible_transitions: Tuple[bool]) -> LowLevelAction:
        """
        @param `orientation` {int} - agent's direction
        @param `possible_transitions` {List[bool]} - (N,E,S,W) absolute values
        @return {LowLevelAction}
        """
        # grid_orientation_shift_value = (1 - orientation) % 4
        grid_orientation_shift_value = orientation
        
        oriented_possible_transitions = np.roll(list(possible_transitions), grid_orientation_shift_value)
        oriented_possible_transitions = tuple(list(oriented_possible_transitions))

        can_go_forward = bool(oriented_possible_transitions[0])
        can_go_right = bool(oriented_possible_transitions[1])
        can_go_back = bool(oriented_possible_transitions[2])
        can_go_left = bool(oriented_possible_transitions[3])

        n_transitions_allowed = int(
            np.sum([int(can_go_left),
                    int(can_go_forward),
                    int(can_go_right),
                    int(can_go_back)])
        )

        ### {STOP} action is mapped 1:1 between High and Low levels.
        if self == self.STOP:
            return LowLevelAction.STOP_MOVING

        ### {STOP} if there are no transitions allowed.
        if n_transitions_allowed < 1:
            return LowLevelAction.DO_NOTHING

        #

        ### 1 TRANSITION ALLOWED
        if n_transitions_allowed == 1:
            if can_go_left is True:
                return LowLevelAction.MOVE_LEFT
            if can_go_forward is True:
                return LowLevelAction.MOVE_FORWARD
            if can_go_right is True:
                return LowLevelAction.MOVE_RIGHT
            if can_go_back is True:
                return LowLevelAction.DO_NOTHING

        ### 2 TRANSITION ALLOWED (LEFT)
        if n_transitions_allowed == 2 and self == self.LEFT_ORIENTED:
            if can_go_left is True:
                return LowLevelAction.MOVE_LEFT
            if can_go_forward is True:
                return LowLevelAction.MOVE_FORWARD

        ### 2 TRANSITION ALLOWED (RIGHT)
        if n_transitions_allowed == 2 and self == self.RIGHT_ORIENTED:
            if can_go_right is True:
                return LowLevelAction.MOVE_RIGHT
            if can_go_forward is True:
                return LowLevelAction.MOVE_FORWARD

        #

        if n_transitions_allowed == 2:
            print()
            print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")
            print(" HighLevelAction.to_low_level(): case not supported.")
            print("  > orientation = ", orientation)
            print("  > possible_transitions = ", possible_transitions)
            print("  > possible_transitions = ", possible_transitions)
            print("  > oriented_possible_transitions = ", oriented_possible_transitions)
            print("  > can_go_forward = ", can_go_forward)
            print("  > can_go_right = ", can_go_right)
            print("  > can_go_back = ", can_go_back)
            print("  > can_go_left = ", can_go_left)
            print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ")
            print()
            return LowLevelAction.DO_NOTHING

        raise Exception('{N_TRANSITIONS_ALLOWED} could not be greater than 2.')
