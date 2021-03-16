import time

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.observations import LocalObsForRailEnv
from flatland.utils.rendertools import RenderTool

import configs as Configs

###


class Environment():
    def __init__(self):
        self._env = None
        self._observator = None
        self._emulator = None

        self.initialize()

    def initialize(self):
        self._observator = TreeObsForRailEnv(max_depth=2)

        self._env = RailEnv(
            width=Configs.WINDOW_WIDTH,
            height=Configs.WINDOW_HEIGHT,
            number_of_agents=Configs.NUMBER_OF_AGENTS,
            rail_generator=random_rail_generator(),
            obs_builder_object=self._observator,
        )

        self._emulator = RenderTool(self._env)

    def perform_actions(self, actions):
        next_obs, all_rewards, done, _ = self._env.step(actions)
        return next_obs, all_rewards, done, _

    def render(self):
        self._emulator.render_env(show=True, show_observations=True, show_predictions=False)

    def reset(self):
        self._emulator.reset()

        observations, info = self._env.reset()
        return observations, info

    def sleep(self, seconds=0.3):
        time.sleep(seconds)

    def finished(self, done=None):
        return done is not None and done['__all__'] is True
