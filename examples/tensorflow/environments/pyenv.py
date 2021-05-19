from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.utils.rendertools import RenderTool
from flatland.utils.rendertools import AgentRenderVariant

import configs as Configs

from observators.binary_tree import BinaryTreeObservator

###


class FlatlandEnvironmentSingleAgent(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()

        self._state = None

        self._reward_spec = None
        self._action_spec = None
        self._discount_spec = None
        self._step_type_spec = None
        self._observation_spec = None

        # flatland
        self._env = None
        self._emulator = None
        self._observator = None

        self.initialize()

    def initialize(self):
        self._observator = BinaryTreeObservator(max_memory=Configs.OBS_TREE_N_NODES)

        self._env = RailEnv(
            width=Configs.RAIL_ENV_WIDTH,
            height=Configs.RAIL_ENV_HEIGHT,
            rail_generator=random_rail_generator(),
            # schedule_generator=None,
            number_of_agents=Configs.N_OF_AGENTS,
            obs_builder_object=self._observator,
            # malfunction_generator_and_process_data=None,
            # malfunction_generator=None,
            remove_agents_at_target=True,
            random_seed=Configs.RANDOM_SEED,
            # record_steps=False,
            # close_following=True
        )

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator = RenderTool(
                self._env,
                # gl="PGL",
                # jupyter=False,
                agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                show_debug=True,
                # clear_debug_text=True,
                screen_width=Configs.EMULATOR_WINDOW_WIDTH,
                screen_height=Configs.EMULATOR_WINDOW_HEIGHT,
            )

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
                shape=(Configs.OBS_TREE_NODE_N_FEATURES,),
                dtype=np.int32,
                name='observation',
            )

        return self._observation_spec

    ###

    def _is_episode_ended(self, done):
        return type(done) is dict and done['__all__'] is True

    def _reset(self):
        '''
        Starts a new sequence, returns the first `TimeStep` of this sequence. \n
        See `reset(self)` docstring for more details.
        '''
        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.reset()

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
        _, _, done, _ = self._env.step({ 0: action })
        # observations, rewards, done, info = self._env.step(action)
        # self._state = (observations, info)

        self._state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int32)

        # if self._is_episode_ended(done):
        #     # The last action ended the episode.
        #     # Ignore the current action and start a new episode.
        #     return self.reset()

        self.render(mode='human')

        if self._is_episode_ended(done):
            return ts.termination(self._state, reward=1)
        else:
            return ts.transition(self._state, reward=1, discount=.8)

    ###

    def render(self, mode='rgb_array'):
        if Configs.EMULATOR_ACTIVE is True and mode == 'human':
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)