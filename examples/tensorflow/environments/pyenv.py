from __future__ import absolute_import, division, print_function

import configs as Configs
import numpy as np
from libs.rail_env import FlatlandRailEnv
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

###


class FlatlandEnvironmentSingleAgent(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()

        # flatland
        self._env = None

        self._state = None

        self._reward_spec = None
        self._action_spec = None
        self._discount_spec = None
        self._step_type_spec = None
        self._observation_spec = None

        self.initialize()

    def initialize(self):
        self._env = FlatlandRailEnv()

    ###

    @property
    def batched(self):
        """
        Whether the environment is batched or not. \n
        If the environment supports batched observations and actions, then overwrite
        this property to True. \n
        A batched environment takes in a batched set of actions and returns a
        batched set of observations. This means for all numpy arrays in the input
        and output nested structures, the first dimension is the batch size. \n
        When batched, the left-most dimension is not part of the action_spec
        or the observation_spec and corresponds to the batch dimension. \n
        ### Returns
        A boolean indicating whether the environment is batched or not.
        """
        return False

    def get_info(self):
        raise NotImplementedError('No support of `get_info()` for this environment.')

    def get_state(self):
        raise NotImplementedError('This environment has not implemented `get_state()`.')

    def set_state(self, state):
        raise NotImplementedError('This environment has not implemented `set_state()`.')

    def action_spec(self):
        '''
        Defines the actions that should be provided to `step()`. \n
        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values. \n
        ### Returns
        An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        '''
        if self._action_spec is None:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(),
                dtype=np.int32,
                minimum=0,
                maximum=4,
                name='action',
            )

        return self._action_spec

    def observation_spec(self):
        '''
        Defines the observations provided by the environment. \n
        May use a subclass of `ArraySpec` that specifies additional
        properties such as min and max bounds on the values. \n
        ### Returns
        An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        '''
        if self._observation_spec is None:
            self._observation_spec = array_spec.ArraySpec(
                shape=(Configs.OBS_TREE_NODE_N_FEATURES, ),
                dtype=np.int32,
                name='observation',
            )

        return self._observation_spec

    ###

    def _reset(self):
        '''
        Starts a new sequence, returns the first `TimeStep` of this sequence. \n
        See `reset(self)` docstring for more details.
        '''
        # observations, info = self._env.reset()
        _, _ = self._env.reset()

        # self._state = (observations, info)
        # self._state = (observations, rewards, done, info)
        self._state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32)

        return ts.restart(self._state, batch_size=self.batch_size)

    def _step(self, action):
        '''
        Updates the environment according to action and returns a `TimeStep`. \n
        See `step(self, action)` docstring for more details. \n
        ### Arguments
        - action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        '''
        _, _, done, _ = self._env.step({0: action})
        # observations, rewards, done, info = self._env.step(action)
        # self._state = (observations, info)

        self._state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32)

        # if self._is_episode_ended(done):
        #     # The last action ended the episode.
        #     # Ignore the current action and start a new episode.
        #     return self.reset()

        if self._env.episode_finished(done):
            return ts.termination(self._state, reward=1)
        else:
            return ts.transition(self._state, reward=1, discount=.8)
