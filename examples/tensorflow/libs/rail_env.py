import time
from typing import Dict

import configs as Configs
from observators.binary_tree import BinaryTreeObservator
from schemes.action import Action

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

###


class FlatlandRailEnv():
    def __init__(self):
        self._env = None
        self._emulator = None
        self._observator = None

        self.initialize()

    def initialize(self):
        self._observator = BinaryTreeObservator(max_memory=Configs.OBS_TREE_N_NODES)

        self._env = RailEnv(
            width=Configs.RAIL_ENV_WIDTH,
            height=Configs.RAIL_ENV_HEIGHT,
            rail_generator=random_rail_generator(),
            # schedule_generator=None,
            number_of_agents=Configs.N_OF_AGENTS,
            obs_builder_object=self._observator,
            # malfunction_generator_and_process_data=None,
            # malfunction_generator=None,
            remove_agents_at_target=True,
            random_seed=Configs.RANDOM_SEED,
            # record_steps=False,
            # close_following=True
        )

        if Configs.EMULATOR_ACTIVE is True:
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

    ###

    def episode_finished(self, done: Dict):
        return dict is not None and isinstance(done, dict) and done['__all__'] is True

    def reset(self):
        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.reset()

        observations, info = self._env.reset()

        return observations, info

    def step(self, actions: Dict[int, Action]):
        observations, rewards, done, info = self._env.step(actions)

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)

        return observations, rewards, done, info
