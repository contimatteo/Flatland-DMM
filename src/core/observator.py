from copy import deepcopy
from typing import Optional, List, Dict

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.utils.ordered_set import OrderedSet

from configs import configurator as Configs
from utils.obs_node import Node

###


class MaxNodeMemory(Exception):
    pass


###


# same as TreeObsForRailEnv, but exploring in breadth first
# instead of having a max_depth, a max_memory attribute is defined
# max_memory will not count 'inf' nodes, which are STILL CREATED
class BinaryTreeObservator(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """

    tree_explored_actions_char = ['F', 'T']  # F: forward; T: turn

    def __init__(self, max_memory: int, predictor: PredictionBuilder = None):
        super().__init__()
        self.max_memory = max_memory
        self.predictor = predictor
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent = None

    def reset(self):
        self.location_has_target = {
            tuple(agent.target): 1
            for agent in self.env.agents
        }  # dictionary of target positions

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and \
                    _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)
                                                    ] = _agent.malfunction_data['malfunction']

            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                    _agent.initial_position:
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

        observations = super().get_many(handles)

        return observations

    def get(self, handle: int = 0) -> Node:
        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        elif agent.status == RailAgentStatus.DONE_REMOVED and agent.old_position:
            agent_virtual_position = agent.target
            agent.status = RailAgentStatus.DONE
        else:
            # agent.initial_position != agent.target and agent.status == RailAgentStatus.DONE_REMOVED
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!STOPPING SEARCH!!!!!!!!!!!!!!!!!!!!!!!!!')
            # print('agent status:', agent.status)
            # print('agent initial pos:', agent.position, '\ntarget position:', agent.target)
            return None

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()

        # storing all possible transitions
        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        root_node_observation = Node(
            dist_own_target_encountered=0,
            dist_other_target_encountered=0,
            dist_other_agent_encountered=0,
            dist_potential_conflict=0,
            dist_unusable_switch=0,
            tot_unusable_switch=0,
            dist_to_next_branch=0,
            dist_min_to_target=distance_map[(handle, *agent_virtual_position, agent.direction)],
            target_reached=0,
            num_agents_same_direction=0,
            num_agents_opposite_direction=0,
            num_agents_malfunctioning=agent.malfunction_data['malfunction'],
            speed_min_fractional=agent.speed_data['speed'],
            num_agents_ready_to_depart=0,
            pos_x=agent_virtual_position[0],
            pos_y=agent_virtual_position[1],

        )

        # if in straight, the first node is not saved
        if num_transitions > 1:
            queue = [(root_node_observation, agent_virtual_position, agent.direction)]
        else:
            first_node, position, orientation = self._explore_branch(
                handle, agent_virtual_position, agent.direction, 1
            )
            root_node_observation.right_child = first_node
            queue = [(first_node, position, orientation)]

        n_added_nodes = 1
        try:
            while queue:  # I stop when I raise MaxNodeMemory or I expanded all the tree
                n_tuple = queue.pop(0)
                node = n_tuple[0]
                position = n_tuple[1]
                orientation = n_tuple[2]

                # getting all possible transitions and counting them
                possible_transitions = self.env.rail.get_transitions(*position, orientation)
                num_transitions = np.count_nonzero(possible_transitions)

                # we may have a turn without a switch: our orientation changes even if we must go straight forward
                if num_transitions == 1:
                    orientation = np.argmax(possible_transitions)

                direction_to_node_pos = {'left': None, 'right': None}
                for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(0, 4)]):

                    if possible_transitions[branch_direction]:
                        if i==0:
                            direction_to_node_pos['right'] = i
                        elif i==1:
                            direction_to_node_pos['left'] = direction_to_node_pos['right']
                            direction_to_node_pos['right'] = i
                        elif i==3:
                            direction_to_node_pos['left'] = i

                # enumerating direction with respect to [forward, right, back, left]
                for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(0, 4)]):

                    if possible_transitions[branch_direction]:
                        # print('\nfollowing direction:', branch_direction)
                        new_cell = get_new_position(position, branch_direction)

                        # main change w.r.t. TreeObsForRailEnv: breadth first search
                        node_observed, observed_pos, observed_dir = self._explore_branch(
                            handle, new_cell, branch_direction, 1
                        )

                        # check if node_observed is forward
                        if i == direction_to_node_pos.get('left', None):
                            node.left_child = node_observed
                        # check if node_observed is right
                        elif i == direction_to_node_pos.get('right', None):
                            node.right_child = node_observed

                        # print(1)

                        if not node_observed:
                            pass
                            # print(
                            #     '!!!!!!!!!!!!!!!!!!!!!!!!!OBSERVED NONE NODE!!!!!!!!!!!!!!!!!!!!!!!!!'
                            # )
                        n_added_nodes += 1

                        # check if we reached self.max_memory
                        if n_added_nodes >= self.max_memory:
                            raise MaxNodeMemory

                        # if we have still space we append observed node to queue
                        queue.append((node_observed, observed_pos, observed_dir))


            # outside while
            # print('outside while without raising an error\nnumber of added nodes:', n_added_nodes)

        except MaxNodeMemory:
            pass
        else:
            if not queue:
                pass
                # print('!!!!!!!!!expanded all the tree!!!!!!!!!\n')

        return root_node_observation

    def _explore_branch(self, handle, position, direction, tot_dist):

        exploring = True
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_target = False

        visited = OrderedSet()
        agent = self.env.agents[handle]
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = 0
        tot_unusable_switch = 0
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0

        while exploring:
            while exploring:
                # #############################
                # #############################
                # Modify here to compute any useful data required to build the end node's features. This code is called
                # for each cell visited between the previous branching node and the next switch / target / dead-end.
                if position in self.location_has_agent:
                    # print('\n\nLOCATION HAS AGENT\n\n')
                    # exploring = False
                    if tot_dist < other_agent_encountered:
                        other_agent_encountered = tot_dist

                    # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                    if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                        malfunctioning_agent = self.location_has_agent_malfunction[position]

                    other_agent_ready_to_depart_encountered += self.location_has_agent_ready_to_depart.get(
                        position, 0
                    )

                    if self.location_has_agent_direction[position] == direction:
                        # Cummulate the number of agents on branch with same direction
                        other_agent_same_direction += 1

                        # Check fractional speed of agents
                        current_fractional_speed = self.location_has_agent_speed[position]
                        if current_fractional_speed < min_fractional_speed:
                            min_fractional_speed = current_fractional_speed

                    else:
                        # If no agent in the same direction was found all agents in that position are other direction
                        # Attention this counts to many agents as a few might be going off on a switch.
                        other_agent_opposite_direction += self.location_has_agent[position]
                        # exploring = False
                        # print('\n\nOTHER AGENT OPPOSITE DIRECTION\n\n')

                # Check number of possible transitions for agent and total number of transitions in cell (type)
                cell_transitions = self.env.rail.get_transitions(*position, direction)
                transition_bit = bin(self.env.rail.get_full_transitions(*position))
                total_transitions = transition_bit.count("1")
                crossing_found = False
                if int(transition_bit, 2) == int('1000010000100001', 2):
                    crossing_found = True

                # Register possible future conflict
                predicted_time = int(tot_dist * time_per_cell)
                if self.predictor and predicted_time < self.max_prediction_depth:
                    int_position = coordinate_to_position(self.env.width, [position])
                    if tot_dist < self.max_prediction_depth:

                        pre_step = max(0, predicted_time - 1)
                        post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                        # Look for conflicting paths at distance tot_dist
                        if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                            conflicting_agent = np.where(
                                self.predicted_pos[predicted_time] == int_position
                            )
                            for ca in conflicting_agent[0]:
                                if direction != self.predicted_dir[predicted_time][
                                    ca] and cell_transitions[self._reverse_dir(
                                        self.predicted_dir[predicted_time][ca]
                                    )] == 1 and tot_dist < potential_conflict:
                                    potential_conflict = tot_dist
                                if self.env.agents[
                                    ca
                                ].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                    potential_conflict = tot_dist

                        # Look for conflicting paths at distance num_step-1
                        elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                            conflicting_agent = np.where(
                                self.predicted_pos[pre_step] == int_position
                            )
                            for ca in conflicting_agent[0]:
                                if direction != self.predicted_dir[pre_step][ca] \
                                        and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                        and tot_dist < potential_conflict:  # noqa: E125
                                    potential_conflict = tot_dist
                                if self.env.agents[
                                    ca
                                ].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                    potential_conflict = tot_dist

                        # Look for conflicting paths at distance num_step+1
                        elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                            conflicting_agent = np.where(
                                self.predicted_pos[post_step] == int_position
                            )
                            for ca in conflicting_agent[0]:
                                if direction != self.predicted_dir[post_step][ca] and cell_transitions[
                                    self._reverse_dir(
                                        self.predicted_dir[post_step][ca])] == 1 \
                                        and tot_dist < potential_conflict:  # noqa: E125
                                    potential_conflict = tot_dist
                                if self.env.agents[
                                    ca
                                ].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                    potential_conflict = tot_dist

                if position in self.location_has_target and position != agent.target:
                    if tot_dist < other_target_encountered:
                        other_target_encountered = tot_dist

                if position == agent.target and tot_dist < own_target_encountered:
                    own_target_encountered = tot_dist
                    # print('\n\nOWN TARGET ENCOUNTERED\n\n')
                    # exploring = False

                # #############################
                # #############################
                if (position[0], position[1], direction) in visited:
                    last_is_terminal = True
                    # print('\n\n\n\n\n\n\n\n\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ALREADY VISITED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n\n\n\n\n\n\n\n\n\n')
                    # break
                visited.add((position[0], position[1], direction))

                # If the target node is encountered, pick that as node. (TODO Also, no further branching is possible.)
                if np.array_equal(position, self.env.agents[handle].target):
                    last_is_target = True
                    # print('\n\nLAST IS TARGET\n\n')
                    # exploring = False
                    # break

                # Check if crossing is found --> Not an unusable switch
                if crossing_found:
                    # Treat the crossing as a straight rail cell
                    total_transitions = 2
                num_transitions = np.count_nonzero(cell_transitions)

                # Detect Switches that can only be used by other agents.
                if total_transitions > 2 > num_transitions:
                    tot_unusable_switch += 1

                    if other_agent_opposite_direction == 0:
                        unusable_switch += 1

                if num_transitions == 1:
                    # Check if dead-end, or if we can go forward along direction
                    nbits = total_transitions
                    if nbits == 1:
                        # Dead-end!
                        last_is_dead_end = True
                        # print('\n\nDEAD END\n\n')
                        # exploring = False

                    if not last_is_dead_end:
                        # Keep walking through the tree along `direction`
                        exploring = True
                        # convert one-hot encoding to 0,1,2,3
                        direction = np.argmax(cell_transitions)
                        position = get_new_position(position, direction)
                        num_steps += 1
                        tot_dist += 1
                elif num_transitions > 0:
                    # Switch detected
                    last_is_switch = True
                    exploring = False
                    # print('\nADDING SWITCH NODE:', cell_transitions,
                    #       '\nAT POSITION:', position)
                    # break

                elif num_transitions == 0:
                    # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                    # print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                    #      position[1], direction)
                    last_is_terminal = True
                    break

                else:
                    exploring = False
                    # print('TO CHECK WHY IT TERMINATED', flush=True)

            # `position` is either a terminal node or a switch

            # #############################
            # #############################
            # Modify here to append new / different features for each visited cell!

            if last_is_target:
                dist_to_next_branch = tot_dist
                dist_min_to_target = 0
            elif last_is_terminal:
                dist_to_next_branch = np.inf
                # dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
                dist_min_to_target = abs(self.env.agents[handle].target[0] - position[0]) + \
                                     abs(self.env.agents[handle].target[1] - position[1])
            else:
                dist_to_next_branch = tot_dist
                # dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
                dist_min_to_target = abs(self.env.agents[handle].target[0] - position[0]) + \
                                     abs(self.env.agents[handle].target[1] - position[1])

            min_fractional_speed = min_fractional_speed / agent.speed_data['speed']
            if min_fractional_speed > 1: min_fractional_speed = 1

            # TreeObsForRailEnv.Node
            node = Node(
                dist_own_target_encountered=own_target_encountered,
                dist_other_target_encountered=other_target_encountered,
                dist_other_agent_encountered=other_agent_encountered,
                dist_potential_conflict=potential_conflict,
                dist_unusable_switch=unusable_switch,
                dist_to_next_branch=dist_to_next_branch,
                dist_min_to_target=dist_min_to_target,
                num_agents_same_direction=other_agent_same_direction,
                num_agents_opposite_direction=other_agent_opposite_direction,
                num_agents_malfunctioning=malfunctioning_agent,
                speed_min_fractional=min_fractional_speed,
                num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                pos_x=position[0],
                pos_y=position[1],
                tot_unusable_switch=tot_unusable_switch,
                target_reached=int(last_is_target)
            )

            return node, position, direction

    """
    def util_print_obs_subtree(self, tree: Node):

        # Utility function to print tree observations returned by this object.

        self.print_node_features(tree, "root", "")
        for direction in self.tree_explored_actions_char:
            self.print_subtree(tree.childs[direction], direction, "\t")


    @staticmethod
    def print_node_features(node: Node, label, indent):
        print(indent, "Direction ", label, ": ", node.dist_own_target_encountered, ", ",
              node.dist_other_target_encountered, ", ", node.dist_other_agent_encountered, ", ",
              node.dist_potential_conflict, ", ", node.dist_unusable_switch, ", ", node.dist_to_next_branch, ", ",
              node.dist_min_to_target, ", ", node.num_agents_same_direction, ", ", node.num_agents_opposite_direction,
              ", ", node.num_agents_malfunctioning, ", ", node.speed_min_fractional, ", ",
              node.num_agents_ready_to_depart)

    def print_subtree(self, node, label, indent):
        if node == -np.inf or not node:
            print(indent, "Direction ", label, ": -np.inf")
            return

        self.print_node_features(node, label, indent)

        if not node.childs:
            return

        for direction in self.tree_explored_actions_char:
            self.print_subtree(node.childs[direction], direction, indent + "\t")
    """

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)


    def get_observations_len(self) -> int:
        n_nodes = Configs.OBS_TREE_N_NODES
        node_n_features = Node.get_n_of_features()
        return int(node_n_features * n_nodes)
