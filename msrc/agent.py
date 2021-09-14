from tensorforce import Agent


class FLAgent(Agent):

    def act(self, states, internals=None, parallel=0, independent=False, deterministic=True, evaluation=None):

        return super().act(states, internals, parallel, independent, deterministic, evaluation)
