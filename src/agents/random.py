from tf_agents.policies.random_py_policy import RandomPyPolicy

from agents.base import BaseAgent

###


class RandomAgent(BaseAgent):
    def load_policy(self):
        return RandomPyPolicy(time_step_spec=self._time_step_spec, action_spec=self._action_spec)
