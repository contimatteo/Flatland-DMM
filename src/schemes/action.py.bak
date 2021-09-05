import numpy as np

from typing import Tuple
from enum import IntEnum
from flatland.envs.rail_env import RailEnvActions as LowLevelAction

###

# LowLevelAction = RailEnvActions

DO_NOTHING = LowLevelAction.DO_NOTHING
MOVE_LEFT = LowLevelAction.MOVE_LEFT
MOVE_FORWARD = LowLevelAction.MOVE_FORWARD
MOVE_RIGHT = LowLevelAction.MOVE_RIGHT
STOP_MOVING = LowLevelAction.STOP_MOVING

###


class HighLevelAction(IntEnum):
    STOP = 0
    LEFT_ORIENTED = 1
    RIGHT_ORIENTED = 2

    action_map = {LEFT_ORIENTED:MOVE_LEFT, RIGHT_ORIENTED:MOVE_RIGHT}

    def high2low2(self, action, orientation, possible_transitions):

        """
        to change:
        STOP = 1
        LEFT_ORIENTED = 0
        RIGHT_ORIENTED = 2
        """

        if action == self.STOP:
            return STOP_MOVING

        possible_actions = np.roll(possible_transitions, (1-orientation)%4)
        if possible_actions[action]:
            return self.action_map[action]
        else:
            return MOVE_FORWARD


    def high2low(self, action, orientation, possible_transitions):
        """
        possible_transitions is a 4 boolean tuple:
            the 4 indexes correspond to N,E,S,W (absolute directions) and the boolean value to the possibility
            of going in that direction

        action value can be 0=STOP, 1=LEFT_ORIENTED, 2=RIGHT_ORIENTED

        the return is a low level action whose value can be 0=DO_NOTHING, 1=MOVE_LEFT, 2=MOVE_FORWARD,
            3=MOVE_RIGHT, 4=STOP_MOVING

        then:
            1) create a mapping from absolute directions to relative directions (N,E,S,W)->(L,F,R,B)
                which depend on agent orientation
            2) use map 1) and possible_transitions to select only relative directions that can be chosen
                as next action
            3) high level action has only LEFT_ORIENTED or RIGHT_ORIENTED (no forward), then if MOVE_FORWARD
                is an option create a mapping from MOVE_FORWARD to LEFT/RIGHT_ORIENTED, according to the situation
            4) now we have a mapping from all direction the agent can choose and the corresponding high level actions.
                what we need is the reverse of this mapping
            5) add manually all other action mappings (e.g self.STOP:STOP_MOVING)
            6) we have a complete map => apply it to action and return the result
        """
        if action == self.STOP:
            return STOP_MOVING

        # 1)
        directions = {i - 1: (i - orientation) % 4 + 1 for i in range(1, 5)}

        # 2)
        possible_transitions_r = {
            directions.get(d): directions.get(d)
            for d in directions if possible_transitions[d]
        }

        # 3)
        # we have to map forward into left or right depending on the switch (only if the switch has forward possibility)
        if MOVE_FORWARD in possible_transitions_r:
            if MOVE_LEFT in possible_transitions_r:
                # if going left is possible, then forward is mapped into right
                possible_transitions_r[MOVE_FORWARD] = self.RIGHT_ORIENTED
                possible_transitions_r[MOVE_LEFT] = self.LEFT_ORIENTED
            else:
                # if going left is not possible, then forward is mapped into left
                possible_transitions_r[MOVE_FORWARD] = self.LEFT_ORIENTED
                possible_transitions_r[MOVE_RIGHT] = self.RIGHT_ORIENTED
        else:
            possible_transitions_r[MOVE_LEFT] = self.LEFT_ORIENTED
            possible_transitions_r[MOVE_RIGHT] = self.RIGHT_ORIENTED

        # 4)
        action_map = {possible_transitions_r.get(k): k for k in possible_transitions_r}

        # 6)
        return action_map(action)

    ###

    def to_low_level(self, orientation: int, possible_transitions: Tuple[bool]) -> LowLevelAction:
        """
        @param `orientation` {int} - agent's direction
        @param `possible_transitions` {List[bool]} - (N,E,S,W) absolute values
        @return {LowLevelAction}
        """
        oriented_possible_transitions = tuple(
            # list(np.roll(np.array(list(possible_transitions)), -orientation))
            list(np.roll(np.array(list(possible_transitions)), orientation))
        )

        can_go_left = bool(oriented_possible_transitions[3])
        can_go_forward = bool(oriented_possible_transitions[0])
        can_go_right = bool(oriented_possible_transitions[1])
        can_go_back = bool(oriented_possible_transitions[2])

        n_transitions_allowed = int(
            np.sum([int(can_go_left),
                    int(can_go_forward),
                    int(can_go_right),
                    int(can_go_back)])
        )

        ### {STOP} action is mapped 1:1 between High and Low levels.
        if self == self.STOP:
            return STOP_MOVING

        ### {STOP} if there are no transitions allowed.
        if n_transitions_allowed < 1:
            return STOP_MOVING

        #

        ### 1 TRANSITION ALLOWED
        if n_transitions_allowed == 1:
            if can_go_left is True:
                return MOVE_LEFT
            if can_go_forward is True:
                return MOVE_FORWARD
            if can_go_right is True:
                return MOVE_RIGHT
            if can_go_back is True:
                return STOP_MOVING

        ### 2 TRANSITION ALLOWED (LEFT)
        if n_transitions_allowed == 2 and self == self.LEFT_ORIENTED:
            if can_go_left is True:
                return MOVE_LEFT
            if can_go_forward is True:
                return MOVE_FORWARD

        ### 2 TRANSITION ALLOWED (RIGHT)
        if n_transitions_allowed == 2 and self == self.RIGHT_ORIENTED:
            if can_go_right is True:
                return MOVE_RIGHT
            if can_go_forward is True:
                return MOVE_FORWARD

        #

        if n_transitions_allowed == 2:
            raise Exception(
                "HighLevelAction: ({}, {}, {}, {}) case not supported.".format(
                    self, orientation, possible_transitions, oriented_possible_transitions
                )
            )

        raise Exception('{N_TRANSITIONS_ALLOWED} could not be greater than 2.')
