import numpy as np

from flatland.envs.rail_env import RailEnvActions
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

from agents.base import BaseAgent

###


class SimpleAgent(BaseAgent):
    def __init__(self):
        self.model = None
        self.state_size = None

        super().__init__()

    ###

    def __preprocessing(self, current_obs, next_obs):
        return next_obs

    ###

    def initialize(self, state_size, model_instance):
        self.model = model_instance
        self.state_size = state_size

        return self

    def act(self, observation, info):
        """
        - @param observation: input is the observation of the agent
        - @param info: input dict with keys {'action_required', 'malfunction', 'speed', 'status'}
        - @return action: integer related to one specific action
        """
        # if (np.random.rand() < .5):
        #     return self.model.predict(observation)

        return np.random.choice(self.available_actions)

    def step(self, current_obs, action, reward, next_obs, done):
        observation = self.__preprocessing(current_obs, next_obs)

        return

        self.model.remember(observation, action, reward, done, True)
