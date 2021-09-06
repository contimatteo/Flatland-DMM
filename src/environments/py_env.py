import numpy as np

from typing import Dict, Any

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from schemes.action import HighLevelAction

from schemes.node import Node
from utils.environment import RailEnvWrapper

###


class PyEnvironment(py_environment.PyEnvironment):
    def __init__(self, observator):
        super().__init__()

        self.observator = observator

        self._env = RailEnvWrapper(observator=self.observator)

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
            shape=(self.observator.N_FEATURES, ),
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
            maximum=2,  # ISSUE: we need a deep discussion about this
            name='action',
        )

    #

    def _reset(self) -> ts.TimeStep:
        """
        Starts a new sequence, returns the first `TimeStep` of this sequence.

        See `reset(self)` docstring for more details
        """
        observations = self._env.reset()

        observation = Node.dict_to_array(observations)

        return ts.restart(observation, batch_size=self.batch_size)

    def _step(self, action: Dict[int, HighLevelAction]) -> Dict[int, ts.TimeStep]:
        """
        Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        """
        observations, rewards = self._env.step(action)

        if self.is_episode_finished() is True:
            return {}

        time_steps_dict = {}

        for agent_idx in action.keys():
            observation = observations[agent_idx].get_subtree_array()
            reward = rewards[agent_idx]
            discount = .5

            if self._env.get_done()[agent_idx] is True:
                time_step = ts.termination(observation, reward=1)
            else:
                time_step = ts.transition(observation, reward=reward, discount=discount)

            time_steps_dict.update({agent_idx: time_step})

        return time_steps_dict
