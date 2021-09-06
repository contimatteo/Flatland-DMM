import numpy as np

from typing import Dict

###


class Node:
    # we have the same properties given by flatland in the namedtuple
    # only added node_code: check tree_observator (I'm going to update it) for usage
    def __init__(
        self,
        node_code=0,
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
        forward_child=None,
        turn_child=None
    ):
        self.dist_min_to_target = dist_min_to_target
        self.dist_other_agent_encountered = dist_other_agent_encountered
        self.dist_other_target_encountered = dist_other_target_encountered
        self.dist_own_target_encountered = dist_own_target_encountered
        self.dist_potential_conflict = dist_potential_conflict
        self.dist_to_next_branch = dist_to_next_branch
        self.dist_unusable_switch = dist_unusable_switch
        self.forward_child = forward_child
        self.node_code = node_code
        self.num_agents_malfunctioning = num_agents_malfunctioning
        self.num_agents_opposite_direction = num_agents_opposite_direction
        self.num_agents_ready_to_depart = num_agents_ready_to_depart
        self.num_agents_same_direction = num_agents_same_direction
        self.speed_min_fractional = speed_min_fractional
        self.turn_child = turn_child

    #

    @staticmethod
    def dict_to_array(nodes: Dict[int, 'Node']) -> np.ndarray:
        observation = []

        for node_obs in nodes.values():
            observation.append(node_obs.get_subtree_array())

        return np.array(observation).flatten()

    #

    # returns a list of numerical attributes (children nodes excluded)
    def get_attribute_list(self, attr_list=[]):
        # attr_list is supposed to be a list of str (attribute names)
        # if no attr_list is given, all numerical attributes are given, sorted
        if not attr_list:
            attr_list = list(self.__dict__.keys())  # excluding children attributes
            attr_list.remove('forward_child')
            attr_list.remove('turn_child')
            attr_list.sort()  # mantaining always the same order

        return [self.__dict__.get(attr, None) for attr in attr_list]

    # simply returns right and left child
    def get_childs(self):
        return self.forward_child, self.turn_child

    # returns a flattened array of node attributes
    # designed to be the input of the neural network
    def get_subtree_array(self, attr_list=[]):
        # attr_list is supposed to be a list of str (attribute names)
        # if no attr_list is given, all numerical attributes are given
        subtree_list = self.get_attribute_list(attr_list)
        visited = []
        last = [self]
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

        return subtree_array
