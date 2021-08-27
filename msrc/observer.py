from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from msrc import config


class TreeTensorObserver(TreeObsForRailEnv):
    def __init__(self):
        super(TreeTensorObserver, self).__init__(
            max_depth=config.OBS_TREE_DEPTH,
            predictor=ShortestPathPredictorForRailEnv()
        )

    def get(self, handle: int = 0):
        return super(TreeTensorObserver, self).get(handle)
