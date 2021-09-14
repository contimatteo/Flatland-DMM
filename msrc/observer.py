from typing import Optional, List

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from msrc import config


class TreeTensorObserver(TreeObsForRailEnv):
    extra_params_len = 0  # !! to change if additional info params are passed in the node_to_np function

    # OBS SPECIFICATION
    obs_n_features = len(config.OBSERVED_NODE_PARAMS) + extra_params_len
    obs_n_nodes = 2**(config.OBS_TREE_DEPTH + 1) - 1
    obs_spec = dict(
        type='float',
        shape=(config.N_TRAINS, obs_n_nodes, obs_n_features),
        min_value=0.0, max_value=config.OBS_MAX_VALUE
    )

    def __init__(self):
        super(TreeTensorObserver, self).__init__(
            max_depth=config.OBS_TREE_DEPTH, predictor=ShortestPathPredictorForRailEnv()
        )
        self.allowed_directions = {}

    def get(self, handle: int = 0):
        obs = super(TreeTensorObserver, self).get(handle)

        # Save the root's allowed directions to aid the action remapping
        if isinstance(obs, Node):
            dirs = [k for k in obs.childs.keys() if isinstance(obs.childs[k], Node)]
        else:
            dirs = []
        self.allowed_directions[handle] = dirs

        # Return the flattened node
        return self.flatten(self.tree_to_np(obs))

    def get_many(self, handles: Optional[List[int]] = None):
        # Call the super's get many, which automatically builds a dict of observations
        many_obs = super(TreeTensorObserver, self).get_many(handles)
        # Then convert the dictionary to a list and then into a tensor, returning it
        obs_list = list(many_obs.values())
        obs_tensor = np.array(obs_list, dtype=np.float32)
        return obs_tensor

    def tree_to_np(self, root, depth=0):
        # Base case
        if depth > config.OBS_TREE_DEPTH:
            return None

        if isinstance(root, Node):
            # Get root value to array
            np_value = TreeTensorObserver.node_to_np(root)

            # Populate the allowed branches, filling in missing nodes
            branches = list(filter(lambda n: isinstance(n, Node), root.childs.values()))

            if len(branches) > 2:
                raise Exception("Node " + root + "has > 2 children")
            while len(branches) < 2:
                branches.append(float("-inf"))
        else:
            # Missing value, set the value to a zeros array and branches to empty
            # FIXME: maybe instead of zeros use the max value
            np_value = np.zeros(self.obs_n_features)
            branches = [float("-inf"), float("-inf")]

        # Return the tree node
        lx_node = self.tree_to_np(branches[0], depth + 1)
        rx_node = self.tree_to_np(branches[1], depth + 1)
        return {"value": np_value.tolist(), "childs": [lx_node, rx_node]}

    @staticmethod
    def flatten(root):
        # Support function
        def _flatten(node, destination):
            assert node is not None
            destination.append(node["value"])
            for child in node["childs"]:
                if child is not None:
                    _flatten(child, destination)
            return destination

        return np.array(_flatten(root, list()))

    @staticmethod
    def node_to_np(node, extra=None):
        node_array = np.array(
            [node.__getattribute__(param) for param in config.OBSERVED_NODE_PARAMS]
        )
        node_array[node_array == float('inf')] = config.OBS_MAX_VALUE
        if extra is None:
            return node_array
        else:
            return np.concatenate((extra, node_array))
