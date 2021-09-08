import numpy as np

from rl.core import Env, Processor, Space
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from schemes.action import HighLevelAction
from utils.environment import RailEnvWrapper

###


class KerasEnvironment(Env):
    def __init__(self, observator):
        super().__init__()

        self.observator = observator

        self._env = RailEnvWrapper(observator=self.observator)

    #

    def observation_spec(self):
        return array_spec.ArraySpec(
            shape=(self.observator.N_FEATURES, ),
            dtype=np.int32,
            name='observation',
        )

    def action_spec(self):
        raw_actions_values = list(map(int, HighLevelAction))

        return array_spec.BoundedArraySpec(
            name='action',
            dtype=np.int32,
            shape=(self._env.n_agents, ),
            minimum=min(raw_actions_values),
            maximum=max(raw_actions_values),
        )

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())

    #

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def render(self, mode='human', close=False):
        pass

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        #### Returns
        - observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        observations = self._env.reset()
        return observations[0].get_subtree_array()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        observations, rewards = self._env.step(action)

        observation = observations[0].get_subtree_array()
        reward = rewards[0]
        done = self._env.get_done()[0]
        info = self._env.get_info()

        return (observation, reward, done, {})
