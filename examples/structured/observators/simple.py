import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4_utils import get_new_position

###


class SimpleObs(ObservationBuilder):
    """
    Simplest observation builder. The object returns observation vectors with 5 identical components,
    all equal to the ID of the respective agent.
    """
    def reset(self):
        return

    def get(self, handle):
        observation = handle * np.ones(5)
        return observation
