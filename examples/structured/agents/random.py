import numpy as np


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, info):
        """
        :param state: input is the observation of the agent
        :param info: input dict with keys {'action_required', 'malfunction', 'speed', 'status'}
        :return: returns an action
        """
        return np.random.choice(np.arange(self.action_size))

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        observations, action_dict, all_rewards, next_obs, done = memories
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return
