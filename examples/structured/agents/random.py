import numpy as np

from flatland.envs.rail_env import RailEnvActions

###

ACTIONS = set([
    RailEnvActions.DO_NOTHING,
    RailEnvActions.MOVE_LEFT,
    RailEnvActions.MOVE_FORWARD,
    RailEnvActions.MOVE_RIGHT,
    RailEnvActions.STOP_MOVING,
])

###


class Agent:
    def __init__(self, state_size):
        self.state_size = state_size

    def act(self, state, info) -> int:
        """
        - @param state: input is the observation of the agent
        - @param info: input dict with keys {'action_required', 'malfunction', 'speed', 'status'}
        - @return action: integer related to one specific action
        """

        return np.random.choice(list(ACTIONS))

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        - @param memories: SARS Tuple to be
        """

        observations, action_dict, all_rewards, next_obs, done = memories

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return
