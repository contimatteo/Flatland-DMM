from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
# from tf_agents.environments import tf_py_environment
# from tf_agents.environments import utils
from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.utils.rendertools import RenderTool
from flatland.utils.rendertools import AgentRenderVariant

import configs as Configs

from observators.binary_tree import BinaryTreeObservator

###


class Flatland(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()

        self._action_spec = np.arange(Configs.NUMBER_OF_AGENTS, dtype=np.int32)
        self._observation_spec = np.arange(Configs.OBS_TREE_N_FEATURES, dtype=np.int32)

        self._state = None

        # flatland
        self._env = None
        self._emulator = None
        self._observator = None

        self.initialize()

    def initialize(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, ), dtype=np.int32, minimum=0, name='observation'
        )

        self._observator = BinaryTreeObservator(max_memory=Configs.OBS_TREE_N_NODES)

        self._env = RailEnv(
            width=Configs.RAIL_ENV_WIDTH,
            height=Configs.RAIL_ENV_HEIGHT,
            rail_generator=random_rail_generator(),
            # schedule_generator=None,
            number_of_agents=Configs.NUMBER_OF_AGENTS,
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

    # TODO: implementation mising
    @property
    def batched(self):
        # '''
        # Whether the environment is batched or not. \n
        # If the environment supports batched observations and actions, then overwrite
        # this property to True. \n
        # A batched environment takes in a batched set of actions and returns a
        # batched set of observations. This means for all numpy arrays in the input
        # and output nested structures, the first dimension is the batch size. \n
        # When batched, the left-most dimension is not part of the action_spec
        # or the observation_spec and corresponds to the batch dimension. \n
        # ### Returns
        # A boolean indicating whether the environment is batched or not.
        # '''
        return False

    # TODO: implementation mising
    @property
    def batch_size(self):
        # '''
        # The batch size of the environment. \n
        # ### Returns
        # The batch size of the environment, or `None` if the environment is not
        # batched. \n
        # ### Raises
        # RuntimeError: If a subclass overrode batched to return True but did not
        #     override the batch_size property.
        # '''
        # if self.batched:
        #     raise RuntimeError(
        #         'Environment %s marked itself as batched but did not override the '
        #         'batch_size property' % type(self)
        #     )
        return None

    def action_spec(self):
        '''
        Defines the actions that should be provided to `step()`. \n
        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values. \n
        ### Returns
        An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        '''
        return self._action_spec

    def observation_spec(self):
        '''
        Defines the observations provided by the environment. \n
        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values. \n
        ### Returns
        An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        '''
        return self._observation_spec

    def get_info(self):
        # '''
        # Returns the environment info returned on the last step. \n
        # ### Returns
        # Info returned by last call to step(). None by default. \n
        # ### Raises
        # NotImplementedError: If the environment does not use info.
        # '''
        (observations, info) = self._state
        return info

    ###

    # TODO: implementation mising
    def _empty_observation(self):
        raise NotImplementedError('implementation for the `empty` observation is missing.')

    def _episode_ended(self, done):
        return done is dict and done['__all__'] is True

    def _reset(self):
        '''
        Starts a new sequence, returns the first `TimeStep` of this sequence. \n
        See `reset(self)` docstring for more details.
        '''
        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.reset()

        # done = None
        # rewards = None
        observations, info = self._env.reset()

        self._state = (observations, info)
        # self._state = (observations, rewards, done, info)

        return ts.restart(np.array([self._state], dtype=np.float32))

    def _step(self, actions):
        '''
        Updates the environment according to action and returns a `TimeStep`. \n
        See `step(self, action)` docstring for more details. \n
        ### Arguments
        - action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        '''

        observations, rewards, done, info = self._env.step(actions)

        self._state = (observations, info)
        transition = np.array([self._state], dtype=np.int32)

        # if self._episode_ended(done):
        #     # The last action ended the episode.
        #     # Ignore the current action and start a new episode.
        #     return self.reset()

        if self._episode_ended(done):
            return ts.termination(transition, rewards)
        else:
            return ts.transition(transition, reward=rewards, discount=1.0)

    ###

    def render(self, mode='rgb_array'):
        if Configs.EMULATOR_ACTIVE is True and mode == 'human':
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_MILLISECONDS)

    # TODO: implementation mising
    def seed(self, seed):
        # '''
        # Seeds the environment.
        # ### Arguments
        # - seed: Value to use as seed for the environment.
        # '''
        # del seed  # unused
        raise NotImplementedError('No seed support for this environment.')
