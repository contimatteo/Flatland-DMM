import inspect
import json
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import configs as Configs

###

RAW_NODE_FEATURES = np.sort(
    [
        "dist_own_target_encountered",
        "dist_other_target_encountered",
        "dist_other_agent_encountered",
        "dist_potential_conflict",
        "dist_unusable_switch",
        "dist_to_next_branch",
        "dist_min_to_target",
        "num_agents_same_direction",
        "num_agents_opposite_direction",
        "num_agents_malfunctioning",
        "num_agents_ready_to_depart",
        "speed_min_fractional",
    ]
)

ENV_N_CELLS = Configs.RAIL_ENV_N_CELLS

OBS_TREE_N_NODES = Configs.OBSERVATOR_TREE_N_NODES
OBS_TREE_N_ATTRIBUTES = Configs.OBSERVATION_TREE_N_ATTRIBUTES
OBS_TREE_STATE_SIZE = Configs.OBSERVATION_TREE_STATE_SIZE

if OBS_TREE_N_ATTRIBUTES - 1 != RAW_NODE_FEATURES.shape[0]:
    raise Exception('OBS_TREE_N_ATTRIBUTES does not match the number of features selected.')

###


class TreeProcessor():
    @staticmethod
    def __is_valid_Node(node):
        return node != float("inf") and node != float("-inf") and type(node).__name__ == 'Node'

    @staticmethod
    def __is_valid_node(node):
        return node != float("inf") and node != float("-inf") and isinstance(node, dict)

    @staticmethod
    def __parse_nodes_recursively(raw_node):
        parsed_node = {'childs': {}}

        for (key, value) in inspect.getmembers(raw_node):
            if key.startswith('_'):
                continue

            elif key == 'childs':
                for direction_key in list(value.keys()):
                    raw_child_node = value[direction_key]

                    if TreeProcessor.__is_valid_Node(raw_child_node):
                        # recursive call
                        child_node = TreeProcessor.__parse_nodes_recursively(raw_child_node)
                        parsed_node['childs'][direction_key] = child_node

            elif key in RAW_NODE_FEATURES:
                parsed_node[key] = value

        return parsed_node

    @staticmethod
    def __obs_to_list(obs, node_height):
        # TODO: change the -1.0 with a correct one
        node_as_list = np.full(OBS_TREE_N_ATTRIBUTES, -1.0)

        node_as_list[0] = node_height
        node_as_list[1:] = [obs[key] for key in RAW_NODE_FEATURES]

        return node_as_list

    @staticmethod
    def __flatten_observation(obs, node_height=0):
        flat_nodes = np.array([])

        # convert root node to list
        root_flat = TreeProcessor.__obs_to_list(obs, node_height)
        # concatenate the generated list
        flat_nodes = np.concatenate((flat_nodes, root_flat), axis=0)

        if 'childs' in list(obs.keys()):
            for direction_key in list(obs['childs'].keys()):
                child = obs['childs'][direction_key]

                if TreeProcessor.__is_valid_node(child):
                    # convert child node to list
                    childs_flat = TreeProcessor.__flatten_observation(child, node_height + 1)
                    # concatenate the generated list
                    flat_nodes = np.concatenate((flat_nodes, childs_flat), axis=0)

        return flat_nodes

    ###

    @staticmethod
    def from_observation_to_nodes_dict(raw_obs):
        """
        - @param raw_obs (Node): tree observation of the environment
        - @return (dict): observation converted to dict
        """
        if not TreeProcessor.__is_valid_Node(raw_obs):
            raise Exception('parameter must be instance of Node class.')

        nodes_as_dict = TreeProcessor.__parse_nodes_recursively(raw_obs)

        return nodes_as_dict

    @staticmethod
    def from_nodes_dict_to_memory_record(obs):
        """
        - @param obs (dict): observation
        - @return (list): observation converted to list
        """
        if not isinstance(obs, dict):
            raise Exception('parameter must be a dict.')

        flatted_obs = TreeProcessor.__flatten_observation(obs)

        if flatted_obs.shape[0] != OBS_TREE_STATE_SIZE:
            raise Exception('somenthing went wrong during the creation of a memory record.')

        return flatted_obs

    @staticmethod
    def remove_infinity_values(record):
        """
        - @param record (list):
        - @return (list): 
        """
        record[record == float("inf")] = ENV_N_CELLS / 2
        record[record == float("-inf")] = 0

        return record

    @staticmethod
    def scale_to_range(record):
        """
        - @param record (list):
        - @param rannge (tuple):
        - @return (list): 
        """
        return MinMaxScaler(feature_range=(0, 1)).fit_transform([record])[0]
