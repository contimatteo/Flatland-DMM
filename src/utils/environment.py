import numpy as np
import time

from typing import Dict, Any, Tuple, List
from flatland.envs.rail_env import RailEnv, EnvAgent, Grid4TransitionsEnum
from flatland.envs.rail_generators import random_rail_generator, sparse_rail_generator
from flatland.utils.rendertools import AgentRenderVariant, RenderTool
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen

import configs as Configs

from schemes.action import HighLevelAction, LowLevelAction
from schemes.node import Node

###


class RailEnvWrapper:
    def __init__(
            self,
            observator,
            rail_generator,
            schedule_generator,
            malfunction_generator):


        self._info = None
        self._done = None

        self._observator = observator
        self.rail_generator = rail_generator
        self.schedule_generator = schedule_generator
        self.malfunction_generator = malfunction_generator

        self._rail_env = RailEnv(
            width=Configs.MAP_WIDTH,
            height=Configs.MAP_HEIGHT,
            rail_generator=self.rail_generator,
            schedule_generator=self.schedule_generator,
            number_of_agents=Configs.N_AGENTS,
            obs_builder_object=self._observator,
            # malfunction_generator_and_process_data=None,
            malfunction_generator=self.malfunction_generator,
            remove_agents_at_target=Configs.REMOVE_AGENTS_AT_TARGET,
            random_seed=Configs.SEED,
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

    def get_grid(self) -> np.ndarray:
        return self._rail_env.rail.grid

    def get_agent(self, agent_index: int) -> EnvAgent:
        return self._rail_env.agents[agent_index]

    def get_agent_position(self, agent_index: int) -> Tuple[int, int]:
        """
        maybe not so easy:
            - if agent.status == READY_TO_DEPART the agent is already asking for observations and answering with
                some decisions, but its position in still None
                ==> in this case it's maybe better to return agent.initial_position
            - we have 2 cases when the agent.position==None (agent.status==READY_TO_DEPART & agent.status==DONE_REMOVED),
                maybe we want to distinguish those

        remember also to not use agent.position during observations (agent.old_position becomes the correct one)
        """
        return self.get_agent(agent_index).position

    def get_agent_direction(self, agent_index: int) -> Grid4TransitionsEnum:
        return self.get_agent(agent_index).direction

    def get_agent_allowed_directions(self, agent_index: int) -> List[bool]:
        position = self.get_agent_position(agent_index)
        orientation = self.get_agent_direction(agent_index)

        if position is None:
            return [False, False, False, False]

        # the following considers also the agent direction (switches allow to turn only from specific directions)
        return self._rail_env.rail.get_transitions(*position, orientation)

    ###

    def reset(self):
        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.reset()

        observations, self._info = self._rail_env.reset()

        return observations

    def step(self, actions: Dict[int, HighLevelAction]) -> Tuple[Dict[int, Node], Dict[int, float]]:
        # TODO: convert high-level actions to low-level actions
        # ...

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
        # print()
        # print("==================================================================================")
        # agent_index = 0
        # print(self.get_agent_position(agent_index))
        # print(self.get_agent_direction(agent_index))
        # print(self.get_agent_allowed_directions(agent_index))
        # print("==================================================================================")
        # print()
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

        # convert high level actions to low level
        low_lev_actions = {i: HighLevelAction(actions.get(i)).to_low_level(self.get_agent_direction(i),
                                                                           self.get_agent_allowed_directions(i))
                          for i in actions}


        observations, rewards, self._done, self._info = self._rail_env.step(low_lev_actions)

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)

        return observations, rewards, self._done, self._info
