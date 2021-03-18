import time
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.observations import LocalObsForRailEnv
from flatland.utils.rendertools import RenderTool
from flatland.utils.rendertools import AgentRenderVariant

import configs as Configs
from utils.observator import SimpleObs
from utils.observator import SingleAgentNavigationObs

###


class Environment():
    def __init__(self):
        self._env = None
        self._observator = None
        self._rail_generator = None
        self._emulator = None

        self.initialize()

    def initialize(self):
        # self._observator = SimpleObs()
        # self._observator = SingleAgentNavigationObs()
        self._observator = TreeObsForRailEnv(max_depth=1)

        self._rail_generator = random_rail_generator()

        self._env = RailEnv(
            width=Configs.RAIL_ENV_WIDTH,
            height=Configs.RAIL_ENV_HEIGHT,
            rail_generator=self._rail_generator,
            # schedule_generator=None,
            number_of_agents=Configs.NUMBER_OF_AGENTS,
            obs_builder_object=self._observator,
            # malfunction_generator_and_process_data=None,
            # malfunction_generator=None,
            remove_agents_at_target=True,
            random_seed=Configs.RANDOM_SEED,
            # record_steps=False,
            # close_following=True
        )

        self._emulator = RenderTool(
            self._env,
            # gl="PGL",
            # jupyter=False,
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=True,
            # clear_debug_text=True,
            screen_width=Configs.EMULATOR_WINDOW_WIDTH,
            screen_height=Configs.EMULATOR_WINDOW_HEIGHT,
        )

    def perform_actions(self, actions):
        """
        - @param actions: Dict[int, RailEnvActions]
        """
        next_obs, all_rewards, done, info = self._env.step(actions)
        return next_obs, all_rewards, done, info

    def render(self, sleep_seconds: float = .5):
        self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
        time.sleep(sleep_seconds)

    def reset(self) -> (dict, dict):
        self._emulator.reset()
        return self._env.reset()

    def get_agents_indexes(self) -> range:
        return range(self._env.get_num_agents())

    def finished(self, done: dict = None) -> bool:
        return done is not None and done['__all__'] is True
