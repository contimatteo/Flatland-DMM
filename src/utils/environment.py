import numpy as np
import time

from typing import Dict, Any, Tuple, List
from flatland.envs.rail_env import RailEnv, EnvAgent, Grid4TransitionsEnum, RailAgentStatus
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

    def get_grid(self) -> np.ndarray:
        return self._rail_env.rail.grid

    def get_agent(self, agent_index: int) -> EnvAgent:
        return self._rail_env.agents[agent_index]

    def get_agent_position(self, agent: EnvAgent) -> Tuple[int, int]:
        """
        maybe not so easy:
        - if agent.status == READY_TO_DEPART the agent is already asking for observations and answering with
          some decisions, but its position in still None
          ==> in this case it's maybe better to return agent.initial_position
        - we have 2 cases when the agent.position==None (agent.status==READY_TO_DEPART & agent.status==DONE_REMOVED),
          maybe we want to distinguish those

        remember also to not use agent.position during observations (agent.old_position becomes the correct one)
        """
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            return agent.initial_position
        elif agent.status == RailAgentStatus.DONE_REMOVED:
            return None  # TODO: reason about this ...
        else:
            return agent.position

    def get_agent_direction(self, agent: EnvAgent) -> Grid4TransitionsEnum:
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            return agent.initial_direction
        elif agent.status == RailAgentStatus.DONE_REMOVED:
            return None  # TODO: reason about this ...
        else:
            return agent.direction

    def get_agent_allowed_directions(self, agent: EnvAgent) -> Tuple[bool]:
        position = self.get_agent_position(agent)
        direction = self.get_agent_direction(agent)

        if position is None or direction is None:
            return [False, False, False, False]

        ### this considers also the agent direction
        ### (switches allow to turn only from specific directions)
        directions = self._rail_env.rail.get_transitions(*position, direction)

        return tuple([x == 1 for x in list(directions)])

    ###

    def reset(self):
        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.reset()

        observations, self._info = self._rail_env.reset()

        return observations

    def step(
        self, high_level_actions: Dict[int, HighLevelAction]
    ) -> Tuple[Dict[int, Node], Dict[int, float]]:
        # TODO: convert high-level actions to low-level actions
        # ...

        low_level_actions = high_level_actions.copy()

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
        # print()
        # print()
        # print("==================================================================================")
        agent: EnvAgent = self.get_agent(0)
        direction = self.get_agent_direction(agent)
        allowed_directions = self.get_agent_allowed_directions(agent)
        high_level_action = low_level_actions[0]
        # print("HIGH-LEVEL action = ", high_level_action, int(high_level_action))
        low_level_action = high_level_action.to_low_level(direction, allowed_directions)
        # print("LOW-LEVEL action = ", low_level_action, int(low_level_action))
        low_level_actions[0] = low_level_action
        # print("==================================================================================")
        # print()
        # print()
        # input("Press Enter to continue...")
        # print()
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

        observations, rewards, self._done, self._info = self._rail_env.step(low_level_actions)

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)

        return observations, rewards
