from typing import Optional, List

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv, Node
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from tf_agents.specs import array_spec

from msrc import config


class TreeTensorObserver(TreeObsForRailEnv):
    extra_params_len = 0  # !! to change if additional info params are passed in the node_to_np function

    def __init__(self):
        super(TreeTensorObserver, self).__init__(
            max_depth=config.OBS_TREE_DEPTH,
            predictor=ShortestPathPredictorForRailEnv()
        )
        # OBS SPECIFICATION
        self.obs_np_len = len(config.OBSERVED_NODE_PARAMS) + TreeTensorObserver.extra_params_len
        self.obs_n_nodes = 2 ** (config.OBS_TREE_DEPTH + 1) - 1
        self.obs_spec = array_spec.BoundedArraySpec(
            shape=(config.N_AGENTS, self.obs_n_nodes, self.obs_np_len),
            dtype=np.float32,
            minimum=0, maximum=config.OBS_MAX_VALUE,
            name='observation'
        )

    def get(self, handle: int = 0):
        obs = super(TreeTensorObserver, self).get(handle)
        return self.flatten(self.tree_to_np(obs))

    def get_many(self, handles: Optional[List[int]] = None):
        many_obs = super(TreeTensorObserver, self).get_many(handles)
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

            # Populate the branches, filling in missing nodes
            branches = list(filter(lambda n: isinstance(n, Node), root.childs.values()))
            if len(branches) > 2:
                raise Exception("Node " + root + "has > 2 children")
            while len(branches) < 2:
                branches.append(float("-inf"))
        else:
            # Missing value, set the value to a zeros array and branches to empty
            # FIXME: maybe instead of zeros use the max value
            np_value = np.zeros(self.obs_np_len)
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
        node_array = np.array([node.__getattribute__(param) for param in config.OBSERVED_NODE_PARAMS])
        node_array[node_array == float('inf')] = config.OBS_MAX_VALUE
        if extra is None:
            return node_array
        else:
            return np.concatenate((extra, node_array))
