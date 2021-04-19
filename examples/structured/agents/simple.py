import numpy as np
import json
import inspect

from flatland.envs.rail_env import RailEnvActions
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

import configs as Configs

from agents.base import BaseAgent
from libs.preprocessor import TreeProcessor

###


class SimpleAgent(BaseAgent):
    def __init__(self):
        self.model = None
        self.state_size = None

        super().__init__()

    ###

    def __preprocessing(self, next_obs):
        parsed_obs = TreeProcessor.from_observation_to_nodes_dict(next_obs)

        memory_record = TreeProcessor.from_nodes_dict_to_memory_record(parsed_obs)

        memory_record = TreeProcessor.remove_infinity_values(memory_record)
        memory_record = TreeProcessor.scale_to_range(memory_record)

        return memory_record

    ###

    def initialize(self, state_size, model_instance):
        self.model = model_instance
        self.state_size = state_size

        return self

    def act(self, current_obs, _, episode):
        """
        - @param current_obs: input is the observation of the agent
        - @param info: input dict with keys {'action_required', 'malfunction', 'speed', 'status'}
        - @return action: integer related to one specific action
        """
        # if episode > 11:
        #     observation = self.__preprocessing(current_obs)
        #     action_key = np.argmax(self.model.predict(observation)[0])
        # else:
        #     action_key = np.random.choice(self.available_actions)

        action_key = np.random.choice(self.available_actions)

        return action_key

    def step(self, current_obs, action, reward, next_obs, done):
        if next_obs is None:
            return

        observation = self.__preprocessing(next_obs)

        self.model.remember(observation, action, reward, done, True)
