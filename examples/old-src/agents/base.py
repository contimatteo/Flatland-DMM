import abc

from tf_agents.trajectories import time_step as ts
from tf_agents.policies import py_policy

from schemes.action import HighLevelAction

###


class BaseAgent(abc.ABC):
    def __init__(self, model, time_step_spec, action_spec):
        self.model = model

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self.policy = self.load_policy()

    ###

    def act(self, time_step) -> HighLevelAction:
        """
        - @param time_step: Tensorflow time-step
        - @return action: integer related to one specific action
        """
        return HighLevelAction(int(self.policy.action(time_step).action))

    def step(self, action: HighLevelAction, time_step: ts.TimeStep, finished: bool):
        if time_step is None:
            return

        reward = time_step.reward
        observation = time_step.observation

        self.model.remember(action, observation, reward, finished, True)

    ###

    @abc.abstractmethod
    def load_policy(self) -> py_policy.Base:
        raise NotImplementedError('`load_policy` method not implemented.')

    # @abc.abstractmethod
    # def save(self, *args, **kwargs):
    #     """
    #     store the current policy
    #     """
    #     raise NotImplementedError('`save` method not implemented.')
