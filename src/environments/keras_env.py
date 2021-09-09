from rl.core import Env, Space

from typing import Dict, Any
from tf_agents.specs import array_spec
import random
import numpy as np
import time

from utils.environment import RailEnvWrapper
from schemes.action import HighLevelAction

import configs as Configs

class ActionSpace(Space):

    def sample(self, seed=None):
        random.seed(seed)
        return random.randint(0,Configs.ACTION_SIZE-1)

    def contains(self, x):
        return x >= 0 and x < Configs.ACTION_SIZE


class KerasEnvironment(Env):
    def __init__(self,
                 observator,
                 rail_generator,
                 schedule_generator,
                 malfunction_generator):

        super().__init__()

        self.observator = observator

        self._env = RailEnvWrapper(observator=observator,
                                   rail_generator=rail_generator,
                                   schedule_generator=schedule_generator,
                                   malfunction_generator=malfunction_generator)

        self.action_space = ActionSpace()

        #

    def is_episode_finished(self) -> bool:
        return self._env.is_episode_finished()

    def get_info(self) -> dict:
        """
        Returns the environment info returned on the last step.
        """
        return self._env.get_info()

    def get_done(self) -> Dict[Any, bool]:
        return self._env.get_done()

    def get_state(self):
        raise NotImplementedError('This environment has not implemented `get_state()`.')

    def set_state(self, _):
        raise NotImplementedError('This environment has not implemented `set_state()`.')

    def observation_spec(self):
        """
        Defines the observations provided by the environment.
        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values.
        Returns:
        An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        """
        return array_spec.ArraySpec(
            shape=(Node.get_n_of_features(),),
            dtype=np.int32,
            name='observation',
        )

    def action_spec(self):
        """
        Defines the actions that should be provided to `step()`.
        May use a subclass of `ArraySpec` that specifies additional properties such
        as min and max bounds on the values.
        Returns:
        An `ArraySpec`, or a nested dict, list or tuple of `ArraySpec`s.
        """
        ### TODO: defines the actions that should be provided to `step()`.

        return array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,  # ISSUE: we need a deep discussion about this
            maximum=Configs.ACTION_SIZE - 1,  # ISSUE: we need a deep discussion about this
            name='action',
        )

    #

    def reset(self):
        """
        Starts a new sequence, returns the first `TimeStep` of this sequence.
        See `reset(self)` docstring for more details
        """
        return self._env.reset()

    def step(self, action: Dict[int, HighLevelAction]):
        """
        Updates the environment according to action and returns a `TimeStep`.
        See `step(self, action)` docstring for more details.
        Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        """

        return self._env.step(action)
    """
            if self.is_episode_finished() is True:
                return {}

            time_steps_dict = {}

            for agent_idx in action.keys():
                observation = observations[agent_idx]
                reward = rewards[agent_idx]
                discount = .5

                if self._env.get_done()[agent_idx] is True:
                    time_step = ts.termination(observation, reward=1)
                else:
                    time_step = ts.transition(observation, reward=reward, discount=discount)

                time_steps_dict.update({agent_idx: time_step})

            return time_steps_dict
    """

    def render(self, mode='human', close=False):
        self._env._emulator.render_env(show=True, show_observations=True, show_predictions=False)
        time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)