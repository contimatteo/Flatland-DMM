import abc

###


class BaseAgent:
    def __init__(self, model, time_step_spec, action_spec):
        self.model = model

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self.policy = self.load_policy()

    ###

    def act(self, time_step):
        """
        - @param time_step: Tensorflow time-step
        - @return action: integer related to one specific action
        """
        return self.policy.action(time_step).action

    def step(self, action, time_step):
        if time_step is None:
            return

        # self.model.remember(observation, action, reward, done, True)

    ###

    @abc.abstractmethod
    def load_policy(self):
        raise NotImplementedError('`load_policy` method not implemented.')

    # @abc.abstractmethod
    # def save(self, *args, **kwargs):
    #     """
    #     store the current policy
    #     """
    #     raise NotImplementedError('`save` method not implemented.')
