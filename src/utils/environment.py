import time
import numpy as np

from typing import Dict, Any, Tuple, List

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv, EnvAgent, Grid4TransitionsEnum, RailAgentStatus
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

import configs as Configs

from schemes.action import HighLevelAction
from schemes.node import Node

###


class RailEnvWrapper:
    def __init__(self, observator, rail_generator, schedule_generator, malfunction_generator):
        self._info = None
        self._done = None

        self._observator = observator
        self._rail_generator = rail_generator
        self._schedule_generator = schedule_generator
        self._malfunction_generator = malfunction_generator

        self._rail_env = RailEnv(
            width=Configs.RAIL_ENV_MAP_WIDTH,
            height=Configs.RAIL_ENV_MAP_HEIGHT,
            rail_generator=self._rail_generator,
            schedule_generator=self._schedule_generator,
            number_of_agents=Configs.N_AGENTS,
            obs_builder_object=self._observator,
            # malfunction_generator_and_process_data=None,
            malfunction_generator=self._malfunction_generator,
            remove_agents_at_target=Configs.RAIL_ENV_REMOVE_AGENTS_AT_TARGET,
            random_seed=Configs.APP_SEED,
            # record_steps=False,
            # close_following=True
        )

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator = RenderTool(
                self._rail_env,
                show_debug=Configs.APP_DEBUG,
                screen_width=Configs.EMULATOR_WINDOW_WIDTH,
                screen_height=Configs.EMULATOR_WINDOW_HEIGHT,
                agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            )

    ###

    def is_episode_finished(self) -> bool:
        return dict is not None and isinstance(self._done, dict) and self._done['__all__'] is True

    def get_info(self) -> dict:

        self._info['action_required2'] = {agent_id: self.action_required(agent_id) for agent_id in range(self._rail_env.get_num_agents())}

        return self._info

    def get_done(self) -> Dict[Any, bool]:
        return self._done

    ###

    @property
    def n_agents(self) -> int:
        return Configs.N_AGENTS

    def get_grid(self) -> np.ndarray:
        return self._rail_env.rail.grid

    def get_agent(self, agent_index: int) -> EnvAgent:
        return self._rail_env.agents[agent_index]

    def get_agent_position(self, agent: EnvAgent) -> Tuple[int, int]:
        """
        maybe not so easy:
            - if agent.status == READY_TO_DEPART the agent is already asking for observations
              and answering with some decisions, but its position in still None
              ==> in this case it's maybe better to return agent.initial_position
            - we have 2 cases when the agent.position==None (agent.status==READY_TO_DEPART & 
              & agent.status==DONE_REMOVED), maybe we want to distinguish those
        (remember also to not use agent.position during observations (agent.old_position becomes the correct one))
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

    def get_agent_transitions(self, agent: EnvAgent) -> Tuple[bool]:
        position = self.get_agent_position(agent)
        direction = self.get_agent_direction(agent)

        if position is None or direction is None:
            return [False, False, False, False]

        ### this considers also the agent direction
        transitions = self._rail_env.rail.get_transitions(*position, direction)

        return tuple([x == 1 for x in list(transitions)])

    ###

    def reset(self):
        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.reset()

        observations, self._info = self._rail_env.reset()

        return observations

    def step(self, high_actions: Dict[int, int]) -> Tuple[Dict[int, Node], Dict[int, float]]:
        low_actions = {}

        for (agent_idx, high_action) in high_actions.items():
            high_action = HighLevelAction(high_action)
            agent = self.get_agent(agent_idx)
            direction = self.get_agent_direction(agent)
            transitions = self.get_agent_transitions(agent)
            low_action = high_action.to_low_level(direction, transitions)
            low_actions.update({agent_idx: low_action})

        observations, rewards, self._done, self._info = self._rail_env.step(low_actions)

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)

        return observations, rewards, self._done, self._info

    def action_required(self, idx_agent):

        get_transitions = self._rail_env.rail.get_transitions

        agent = self.get_agent(idx_agent)
        pos = self.get_agent_position(agent)
        dir = self.get_agent_direction(agent)

        t = get_transitions(*pos, dir)

        # if more than one transition possible we are in switch
        if np.count_nonzero(t) > 1:
            return True

        # if here, then we are in a straight cell
        # check if next is a switch

        dir = t.index(1)
        pos = get_new_position(pos, dir)

        t = get_transitions(*pos, dir)

        # if more than one transition possible we are in switch
        if np.count_nonzero(t) > 1:
            return True


        return False
