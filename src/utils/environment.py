from copy import deepcopy
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
            # record_steps=False,
            # close_following=True
        )

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator = RenderTool(
                self._rail_env,
                show_debug=Configs.DEBUG,
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
        self._rail_env._max_episode_steps = None

        self._info['action_required2'] = {
            agent_id: self.action_required(agent_id)
            for agent_id in range(self._rail_env.get_num_agents())
        }

        obs = {
            agent_id: observations.get(agent_id).get_subtree_array()
            for agent_id in observations
        }

        return obs

    def step(self, high_actions: Dict[int, int]) -> Tuple[Dict[int, Node], Dict[int, float]]:
        low_actions = self.processor_action(high_actions)

        observations, rewards, done, info = self._rail_env.step(low_actions)

        self._done = deepcopy(done)

        observations, rewards, self._info = self.processor_step(observations, info)

        if Configs.EMULATOR_ACTIVE is True:
            self._emulator.render_env(show=True, show_observations=True, show_predictions=False)
            time.sleep(Configs.EMULATOR_STEP_TIMEBREAK_SECONDS)

        return observations, rewards, self._done, self._info

    def processor_step(self, obs, info, attr_list=[]):
        rewards = {}
        for agent_id in range(len(obs)):
            obs_node = obs.get(agent_id)
            TARGET_MASS = 1000

            if obs_node is None:
                rewards[agent_id] = TARGET_MASS * 2
                continue

            ########################## OBSERVATION PREPARATION ##########################
            # attr_list is supposed to be a list of str (attribute names)
            # only the first node is supposed to have only one child
            if not obs_node.left_child:
                assert obs_node.right_child is not None
                first_val = [1]
                subtree_list = [obs_node.right_child.get_attribute_dict(attr_list)]
                last = [obs_node.right_child]
            else:
                first_val = [0]
                subtree_list = [obs_node.get_attribute_dict(attr_list)]
                last = [obs_node]

            ############################ REWARD PREPARATION ############################

            reward = 0
            AGENT_MASS = 1
            MAX_REWARD = 2 * TARGET_MASS
            agent = self.get_agent(agent_id)

            if agent.status == RailAgentStatus.DONE:
                reward += TARGET_MASS * 2

            p = (obs_node.pos_x, obs_node.pos_y)
            t = agent.target

            if last[0].dist_min_to_target == 0:
                attractive_force = TARGET_MASS * 2
            else:
                attractive_force = TARGET_MASS / (
                    last[0].dist_min_to_target * last[0].dist_min_to_target
                )
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
            prob = 1
            while True:
                # the loop is repeated for each depth of tree, starting from 0
                prob /= 2
                for i in range(len(last)):
                    node = last[i]
                    child_list = [
                        child for child in node.get_childs() if child
                    ]  # get_childs() returns forward and turn child even if they are None

                    # observation process
                    l = [child.get_attribute_dict(attr_list) for child in child_list]
                    # l = [attr for child in child_list for attr in child.get_attribute_list(attr_list)]
                    subtree_list += l

                    # reward compute
                    # update attractive force
                    if node.dist_min_to_target == 0:
                        attractive_force += TARGET_MASS * 2 * prob

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

                unusuable_stiches = [
                    unusuable_stiches[i // 2] for i in range(len(unusuable_stiches) * 2)
                ]

            #################################### CONCLUSIVE OBSERVATION TRANSFORMATION / NORMALIZATION
            # transforming into array
            # subtree_array = np.array(subtree_list)

            # removing inf
            # subtree_array[subtree_array == -np.inf] = 0
            # subtree_array[subtree_array == np.inf] = 0
            ########################### NORMALIZATION
            node_list = []
            test_count = 0
            assert len(subtree_list) == Configs.OBS_TREE_N_NODES
            for node in subtree_list:
                normalization_dict = self.get_normalization_dict(node)
                assert len(node) == Node.get_n_of_features()
                for attr in node:
                    test_count += 1
                    if node[attr] == np.inf:
                        node[attr] = normalization_dict[attr]

                    node_list.append(node[attr] / normalization_dict[attr])
                    assert len(node_list) == test_count
                    assert test_count <= (Node.get_n_of_features() * Configs.OBS_TREE_N_NODES)

            assert len(node_list) == (Node.get_n_of_features() * Configs.OBS_TREE_N_NODES)
            # node_list = first_val + node_list  # INFO: @bug

            # if len(node_list) != (Node.get_n_of_features() * Configs.OBS_TREE_N_NODES + 1):
            if len(node_list) != self._observator.get_observations_len():
                print(
                    '\nnumber of node features:', Node.get_n_of_features(),
                    '\nnumber of nodes per obs:', Configs.OBS_TREE_N_NODES, '\nobs len:',
                    len(node_list), '\nexpected len:',
                    # Node.get_n_of_features() * Configs.OBS_TREE_N_NODES + 1
                    Node.get_n_of_features() * Configs.OBS_TREE_N_NODES
                )
            assert len(node_list) == self._observator.get_observations_len()

            obs[agent_id] = np.array(node_list)
            node_list = []

            #################################### CONCLUSIVE REWARD TRANSFORMATION / NORMALIZATION

            if attractive_force > MAX_REWARD:
                attractive_force = MAX_REWARD
            if repulsive_force < - MAX_REWARD:
                repulsive_force = - MAX_REWARD

            reward += attractive_force - repulsive_force
            reward /= MAX_REWARD
            rewards[agent_id] = reward
            assert reward <= 1 and reward >=-1

        info['action_required2'] = {
            agent_id: self.action_required(agent_id)
            for agent_id in range(self._rail_env.get_num_agents())
        }

        return obs, rewards, info

    def get_normalization_dict(self, node_dict):

        branch_length = node_dict.get("dist_to_next_branch") or 1

        max_n_agents = node_dict.get("num_agents_same_direction"
                                     ) + node_dict.get("num_agents_opposite_direction") or 1

        normalization_dict = {
            "dist_own_target_encountered": branch_length,
            "dist_other_target_encountered": branch_length,
            "dist_other_agent_encountered": branch_length,
            "dist_potential_conflict": branch_length,
            "dist_unusable_switch": node_dict.get("tot_unusable_switch") or 1,
            "tot_unusable_switch": branch_length,
            "dist_to_next_branch": Configs.RAIL_ENV_MAP_WIDTH + Configs.RAIL_ENV_MAP_HEIGHT,
            "dist_min_to_target": Configs.RAIL_ENV_MAP_WIDTH + Configs.RAIL_ENV_MAP_HEIGHT,
            "target_reached": 1,
            "num_agents_same_direction": branch_length,
            "num_agents_opposite_direction": branch_length,
            "num_agents_malfunctioning": max_n_agents,
            "speed_min_fractional": 1,
            "num_agents_ready_to_depart": max_n_agents,
            "pos_x": Configs.RAIL_ENV_MAP_WIDTH,
            "pos_y": Configs.RAIL_ENV_MAP_HEIGHT,
        }

        return normalization_dict

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

        if agent.status == RailAgentStatus.DONE:
            return True
        elif agent.status == RailAgentStatus.DONE_REMOVED:
            return False

        pos = self.get_agent_position(agent)
        direction = self.get_agent_direction(agent)

        t = get_transitions(*pos, direction)

        # if more than one transition possible we are in switch
        if np.count_nonzero(t) > 1:
            return True

        # if here, then we are in a straight cell
        # check if next is a switch

        direction = t.index(1)
        pos = get_new_position(pos, direction)

        t = get_transitions(*pos, direction)

        # if more than one transition possible we are in switch
        if np.count_nonzero(t) > 1:
            return True

        return False
