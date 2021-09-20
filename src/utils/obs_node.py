from typing import Dict

import numpy as np

from configs import configurator as Configs

###


class Node:
    # we have the same properties given by flatland in the namedtuple
    # only added node_code: check tree_observator (I'm going to update it) for usage
    def __init__(
        self,
        dist_own_target_encountered=np.inf,
        dist_other_target_encountered=np.inf,
        dist_other_agent_encountered=np.inf,
        dist_potential_conflict=np.inf,
        dist_unusable_switch=np.inf,
        dist_to_next_branch=np.inf,
        dist_min_to_target=np.inf,
        num_agents_same_direction=0,
        num_agents_opposite_direction=0,
        num_agents_malfunctioning=0,
        speed_min_fractional=1.,
        num_agents_ready_to_depart=0,
        pos_x=0,
        pos_y=0,
        left_child=None,
        right_child=None
    ):
        self.dist_min_to_target = dist_min_to_target
        self.dist_other_agent_encountered = dist_other_agent_encountered
        self.dist_other_target_encountered = dist_other_target_encountered
        self.dist_own_target_encountered = dist_own_target_encountered
        self.dist_potential_conflict = dist_potential_conflict
        self.dist_to_next_branch = dist_to_next_branch
        self.dist_unusable_switch = dist_unusable_switch
        self.left_child = left_child
        self.num_agents_malfunctioning = num_agents_malfunctioning
        self.num_agents_opposite_direction = num_agents_opposite_direction
        self.num_agents_ready_to_depart = num_agents_ready_to_depart
        self.num_agents_same_direction = num_agents_same_direction
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.speed_min_fractional = speed_min_fractional
        self.right_child = right_child

    @staticmethod
    def get_n_of_features():
        return len(Node().__dict__) - 2

    def get_attribute_list(self, attr_list=[]):
        # attr_list is supposed to be a list of str (attribute names)
        # if no attr_list is given, all numerical attributes are given, sorted
        if not attr_list:
            attr_list = list(self.__dict__.keys())  # excluding children attributes
            attr_list.remove('left_child')
            attr_list.remove('right_child')
            attr_list.sort()  # mantaining always the same order

        return [self.__dict__.get(attr, None) for attr in attr_list]

    # simply returns right and left child
    def get_childs(self):
        return self.left_child, self.right_child

    # returns a flattened array of node attributes
    # designed to be the input of the neural network
    def get_subtree_array(self, attr_list=[]):
        # attr_list is supposed to be a list of str (attribute names)

        # only the first node is supposed to have only one child
        if not self.left_child:
            assert self.right_child is not None
            subtree_list = [1]
            subtree_list += self.right_child.get_attribute_list(attr_list)
            last = [self.right_child]
        else:
            subtree_list = [0]
            subtree_list += self.get_attribute_list(attr_list)
            last = [self]

        # if no attr_list is given, all numerical attributes are given
        visited = []
        while True:
            for n in last:
                child_list = [
                    child for child in n.get_childs() if child
                ]  # get_childs() returns forward and turn child even if they are None
                l = [attr for child in child_list for attr in child.get_attribute_list(attr_list)]
                subtree_list += l
                visited += child_list

            if not visited:
                break

            last = visited
            visited = []

        # transforming into array
        subtree_array = np.array(subtree_list)

        # removing inf
        subtree_array[subtree_array == -np.inf] = 0
        subtree_array[subtree_array == np.inf] = 0

        if len(subtree_array) != (Node.get_n_of_features() * Configs.OBS_TREE_N_NODES + 1):
            print('\nnumber of node features:', Node.get_n_of_features(),
                  '\nnumber of nodes per obs:', Configs.OBS_TREE_N_NODES,
                  '\nobs len:', len(subtree_array),
                  '\nexpected len:', (Node.get_n_of_features() * Configs.OBS_TREE_N_NODES + 1))
            assert len(subtree_array) == (Node.get_n_of_features() * Configs.OBS_TREE_N_NODES + 1)

        return subtree_array
