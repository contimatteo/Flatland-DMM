import time
from typing import Dict

import configs as Configs
from schemes.action import HighLevelAction

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

###


class RailEnvWrapper:
    def __init__(self, observator):
        self._observator = observator

        self._rail_env = RailEnv(
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
                self._rail_env,
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

        observations, info = self._rail_env.reset()

        return observations, info

    def step(self, actions: Dict[int, HighLevelAction]):
        observations, rewards, done, info = self._rail_env.step(actions)

        # TODO: convert high-level actions to low-level actions
        # ...

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)

        return observations, info, rewards, done
