from gym import Env, spaces

from utils.environment import RailEnvWrapper

###


class KerasEnvironment(Env):
    def __init__(self, observator, rail_generator, schedule_generator, malfunction_generator):
        super().__init__()

        self.observator = observator

        self._env = RailEnvWrapper(
            observator=self.observator,
            rail_generator=rail_generator,
            schedule_generator=schedule_generator,
            malfunction_generator=malfunction_generator
        )

    #

    @property
    def action_space(self):
        return spaces.Discrete(3)

    @property
    def observation_space(self):
        return spaces.Box(low=-1, high=1000, shape=(39, ))

    #

    def seed(self, seed=None):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()
