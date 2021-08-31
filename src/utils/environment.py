import time

from typing import Dict, Any, Tuple, List
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

import configs as Configs

from schemes.action import HighLevelAction
from schemes.node import Node

###


class RailEnvWrapper:
    def __init__(self, observator):
        self._info = None
        self._done = None

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
                agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                show_debug=True,
                screen_width=Configs.EMULATOR_WINDOW_WIDTH,
                screen_height=Configs.EMULATOR_WINDOW_HEIGHT,
            )

    ###

    def is_episode_finished(self) -> bool:
        return dict is not None and isinstance(self._done, dict) and self._done['__all__'] is True

    def get_info(self) -> dict:
        return self._info

    def get_done(self) -> Dict[Any, bool]:
        return self._done

    ###

    def reset(self):
        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.reset()

        observations, self._info = self._rail_env.reset()

        return observations

    def step(self, actions: Dict[int, HighLevelAction]) -> Tuple[Dict[int, Node], Dict[int, float]]:
        # TODO: convert high-level actions to low-level actions
        # ...

        observations, rewards, self._done, self._info = self._rail_env.step(actions)

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)

        return observations, rewards
