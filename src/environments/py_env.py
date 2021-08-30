import numpy as np

from typing import Dict, Any

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from schemes.node import Node
from utils.environment import RailEnvWrapper

###


class PyEnvironment(py_environment.PyEnvironment):
    def __init__(self, observator):
        super().__init__()

        self._info = None
        self._done = None

        self.observator = observator

        self._env = RailEnvWrapper(observator=self.observator)

    #

    def get_info(self):
        """
        Returns the environment info returned on the last step.
        """
        return self._info

    def get_done(self) -> Dict[Any, bool]:
        return self._done

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
        observations, self._info = self._env.reset()

        # TODO: map each observation ({Node} class schema) to its
        # 'flatten' version through the `get_subtree_array() method`.
        observation = [observations[0].get_subtree_array()]

        return ts.restart(observation, batch_size=self.batch_size)

    def _step(self, action) -> ts.TimeStep:
        """
        Updates the environment according to action and returns a `TimeStep`.

        See `step(self, action)` docstring for more details.

        Args:
        action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.
        """
        observations, self._info, rewards, self._done = self._env.step(action)

        # ISSUE: [@contimatteo -> @davidesangiorgi]
        # Sometimes there's not any observation. Why? Is there any way to avoid it?
        if observations[0] is None:
            return ts.termination(None, reward=-1)

        ### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        ### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        ### move this code outside
        observation = []
        for node_obs in observations.values():
            # if node_obs is not None: observation.append(node_obs.get_subtree_array())
            # else: observation.append(np.full(self.observation_spec().shape, -10))
            observation.append(np.full(self.observation_spec().shape, -10))

        observation = np.array(observation).flatten()

        # ISSUE: [@contimatteo -> @davidesangiorgi]
        # some nodes are not converted to an array ...
        for (i, val) in np.ndenumerate(observation):
            if isinstance(val, Node):
                observation[i] = 1  # ISSUE: we need a deep discussion about this

        observation = np.array(observation).flatten()

        ### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        ### # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        if not self._env.episode_finished(self._done):
            rewards = np.array(list(rewards.values()))
            discounts = np.full(rewards.shape, .8)

            return ts.transition(observation, reward=rewards, discount=discounts)
        else:
            return ts.termination(observation, reward=1)
