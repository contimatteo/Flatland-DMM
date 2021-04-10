import numpy as np

from flatland.envs.rail_env import RailEnvActions

from agents.base import BaseAgent

###


class RandomAgent(BaseAgent):
    def __init__(self):
        self.model = None
        self.state_size = None

        super().__init__()

    def initialize(self, state_size, model_instance):
        self.model = model_instance
        self.state_size = state_size

        return self

    def act(self, _, __):
        """
        - @param state: input is the observation of the agent
        - @param info: input dict with keys {'action_required', 'malfunction', 'speed', 'status'}
        - @return action: integer related to one specific action
        """
        return np.random.choice(list(self.available_actions))

    def step(self, current_observation, action, reward, next_observation, done):
        pass
