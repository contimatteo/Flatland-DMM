import numpy as np

###


class Node:
    # we have the same properties given by flatland in the namedtuple
    # only added node_code: check tree_observator (I'm going to update it) for usage
    def __init__(
        self,
        node_code,
        dist_other_target_encountered,
        dist_other_agent_encountered=np.inf,
        dist_potential_conflict=np.inf,
        dist_unusable_switch=np.inf,
        dist_to_next_branch=np.inf,
        dist_min_to_target=np.inf,
        num_agents_same_direction=0,
        num_agents_opposite_direction=0,
        num_agents_malfunctioning=0,
        speed_min_fractional=0,
        num_agents_ready_to_depart=0,
        right_child=None,
        left_child=None
    ):

        self.node_code = node_code
        self.dist_other_target_encountered = dist_other_target_encountered
        self.dist_other_agent_encountered = dist_other_agent_encountered
        self.dist_potential_conflict = dist_potential_conflict
        self.dist_unusable_switch = dist_unusable_switch
        self.dist_to_next_branch = dist_to_next_branch
        self.dist_min_to_target = dist_min_to_target
        self.num_agents_same_direction = num_agents_same_direction
        self.num_agents_opposite_direction = num_agents_opposite_direction
        self.num_agents_malfunctioning = num_agents_malfunctioning
        self.speed_min_fractional = speed_min_fractional
        self.num_agents_ready_to_depart = num_agents_ready_to_depart
        self.right_child = right_child
        self.left_child = left_child

    # returns a list of numerical attributes (children nodes excluded)
    def get_attribute_list(self, attr_list=[]):
        # attr_list is supposed to be a list of str (attribute names)
        # if no attr_list is given, all numerical attributes are given, sorted
        if not attr_list:
            attr_list = list(self.__dict__.keys())  # excluding children attributes
            attr_list.remove('right_child')
            attr_list.remove('left_child')
            attr_list.sort()  # mantaining always the same order
        return [self.__dict__.get(attr, None) for attr in attr_list]

    # simply returns right and left child
    def get_childs(self):
        return self.right_child, self.left_child

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
                child_list = [child for child in n.get_childs() if child]
                l = [attr for child in child_list for attr in child.get_attribute_list(attr_list)]
                subtree_list += l
                visited += child_list

            if not visited:
                break

            last = visited
            visited = []

        return np.array(subtree_list)
