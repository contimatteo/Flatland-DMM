import time
import numpy as np

from typing import Dict, Any, Tuple, List

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv, EnvAgent, Grid4TransitionsEnum, RailAgentStatus
from flatland.utils.rendertools import AgentRenderVariant, RenderTool

from configs import configurator as Configs

from utils.action import HighLevelAction
from utils.obs_node import Node

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
            random_seed=Configs.SEED,
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
        self._info['action_required2'] = {agent_id: self.action_required(agent_id) for agent_id in
                                              range(self._rail_env.get_num_agents())}

        obs = {agent_id: observations.get(agent_id).get_subtree_array() for agent_id in observations}

        return obs

    def step(self, high_actions: Dict[int, int]) -> Tuple[Dict[int, Node], Dict[int, float]]:

        low_actions = self.processor_action(high_actions)

        observations, rewards, self._done, info = self._rail_env.step(low_actions)

        observations, rewards, self._info = self.processor_step(observations, info)

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)

        return observations, rewards, self._done, self._info

    def processor_step(self, obs, info, attr_list=[]):
        rewards = {}
        for agent_id in range(len(obs)):
            ########################## OBSERVATION PREPARATION ##########################
            # attr_list is supposed to be a list of str (attribute names)
            # only the first node is supposed to have only one child
            obs_node = obs.get(agent_id)
            if not obs_node.left_child:
                assert obs_node.right_child is not None
                subtree_list = [1]
                subtree_list += obs_node.right_child.get_attribute_list(attr_list)
                last = [obs_node.right_child]
            else:
                subtree_list = [0]
                subtree_list += obs_node.get_attribute_list(attr_list)
                last = [obs_node]

            ############################ REWARD PREPARATION ############################

            reward = 0
            TARGET_MASS = 1000
            AGENT_MASS = 1
            agent = self.get_agent(agent_id)

            if agent.status == RailAgentStatus.DONE:
                reward += 10000

            p = (obs_node.pos_x, obs_node.pos_y)
            t = agent.target

            attractive_force = 0
            repulsive_force = 0

            unusuable_stiches = [0, 0]

            """
            # compute probability of conflict
            n_opp_l = last[0].left_child.num_agents_opposite_direction
            c_l = sum([math.comb(n_opp_l, k+1) for k in range(n_opp_l)])
            prob_conflict_l = (c_l / (c_l + 1))**last[0].left_child.unusable_switch
            n_opp_r = last[0].right_child.num_agents_opposite_direction
            c_r = sum([math.comb(n_opp_r, k+1) for k in range(n_opp_r)])
            prob_conflict_r = (c_r / (c_r + 1))**last[0].right_child.unusable_switch
            prob_conflict = (prob_conflict_l + prob_conflict_r) / 2
            reward -= prob_conflict
            """

            ############################# NODE EXPLORATION #############################
            # if no attr_list is given, all numerical attributes are given
            visited = []
            count = 0
            while True:
                # the loop is repeated for each depth of tree, starting from 0
                for i in range(len(last)):
                    count += 1
                    node = last[i]
                    child_list = [
                        child for child in node.get_childs() if child
                    ]  # get_childs() returns forward and turn child even if they are None


                    # observation process
                    l = [attr for child in child_list for attr in child.get_attribute_list(attr_list)]
                    subtree_list += l

                    # reward compute
                    # update attractive force
                    if node.dist_min_to_target == 0:
                        attractive_force = TARGET_MASS * 2
                    else:
                        p = (node.pos_x, node.pos_y)
                        dist_to_target = abs(p[0] - t[0]) + abs(p[1] - t[1])
                        attractive_force += TARGET_MASS / (dist_to_target * dist_to_target)

                    # update repulsive force
                    agent_dist = unusuable_stiches[i] + node.dist_unusable_switch
                    if agent_dist == 0:
                        repulsive_force = AGENT_MASS * 100
                    else:
                        tot_mass = node.num_agents_opposite_direction * AGENT_MASS
                        repulsive_force += tot_mass / (agent_dist * agent_dist)

                    # update agent "distances"
                    unusuable_stiches[i] += node.tot_unusable_switch

                    visited += child_list

                if not visited:
                    break

                last = visited
                visited = []

                unusuable_stiches = [unusuable_stiches[i // 2] for i in range(len(unusuable_stiches) * 2)]

            #################################### CONCLUSIVE OBSERVATION TRANSFORMATION / NORMALIZATION
            # transforming into array
            subtree_array = np.array(subtree_list)

            # removing inf
            subtree_array[subtree_array == -np.inf] = 0
            subtree_array[subtree_array == np.inf] = 0

            if len(subtree_array) != (Node.get_n_of_features() * Configs.OBS_TREE_N_NODES + 1):
                print('\nnumber of node features:', Node.get_n_of_features(),
                      '\nnumber of nodes per obs:', Configs.OBS_TREE_N_NODES,
                      '\nobs len:', len(subtree_array),
                      '\nexpected len:', Node.get_n_of_features() * Configs.OBS_TREE_N_NODES + 1)
                assert len(subtree_array) == (Node.get_n_of_features() * Configs.OBS_TREE_N_NODES + 1)

            obs[agent_id] = subtree_array

            #################################### CONCLUSIVE REWARD TRANSFORMATION / NORMALIZATION

            avg_attractive_force = attractive_force / count
            reward += avg_attractive_force - repulsive_force
            rewards[agent_id] = reward

        info['action_required2'] = {agent_id: self.action_required(agent_id) for agent_id in
                                              range(self._rail_env.get_num_agents())}

        return obs, rewards, info



    def processor_action(self, high_actions):
        low_actions = {}

        for (agent_idx, high_action) in high_actions.items():
            high_action = HighLevelAction(high_action)
            agent = self.get_agent(agent_idx)
            direction = self.get_agent_direction(agent)
            transitions = self.get_agent_transitions(agent)
            low_action = high_action.to_low_level(direction, transitions)
            low_actions.update({agent_idx: low_action})

        return low_actions

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
