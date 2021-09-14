import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.utils.rendertools import RenderTool
from tensorforce import Environment

import config
from msrc.observer import TreeTensorObserver


class TFEnvironment(Environment):
    def __init__(self, render=True):
        super(TFEnvironment, self).__init__()

        # RAIL GENERATOR
        self._rail_gen = complex_rail_generator(
            nr_start_goal=config.RAIL_NR_START_GOAL,
            nr_extra=config.RAIL_NR_EXTRA,
            min_dist=config.RAIL_MIN_DIST,
            max_dist=config.RAIL_MAX_DIST,
            seed=config.ENV_SEED
        )

        # OBSERVATION
        self._obs_builder = TreeTensorObserver()

        # ENVIRONMENT
        self._env = RailEnv(
            width=config.ENV_WIDTH,
            height=config.ENV_HEIGHT,
            number_of_agents=config.N_TRAINS,
            remove_agents_at_target=True,
            obs_builder_object=self._obs_builder,
            random_seed=config.ENV_SEED,
            rail_generator=self._rail_gen
        )

        # RENDERER
        self.renderer = RenderTool(
            self._env, show_debug=False, screen_height=1000, screen_width=1000
        ) if render else None

        # ACTION SPEC
        # The input is a 1d int vector of size N_AGENTS, where each number is in 0,1,2
        self._action_spec = dict(type='int', shape=(config.N_TRAINS,), num_values=3)

        # OTHER
        self.info = None
        self.done = False
        self._frame_count = 0

    def states(self):
        """Returns the state space specification"""
        return TreeTensorObserver.obs_spec

    def actions(self):
        """Returns the action space specification"""
        return self._action_spec

    def max_episode_timesteps(self):
        # Note: the max timesteps are defined in the environment construction
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self, num_parallel=None):
        # Reset the environment and internal parameters
        obs, info = self._env.reset(activate_agents=True)
        self.info = info
        self.done = False
        self._frame_count = 0
        # Renderer
        if self.renderer:
            self.renderer.reset()
        return obs

    def execute(self, actions):
        # Convert the pseudo actions <0,1,2> into the real action dictionary {h: <0,...,4>}
        # and pass it to the flatland environment
        action_dict = self._convert_actions(actions)
        obs, reward_dict, done_dict, self.info = self._env.step(action_dict)

        # Reward determination
        # TODO: weight train collisions and other weighting (eg. distance) aka "Reward Shaping"
        reward = sum([self._agent_reward(h) for h in self._env.get_agent_handles()])

        # Rendering
        if self.renderer:
            self.renderer.render_env(show=True, show_observations=False)

        # End of episode determination and time step returning
        self.done = done_dict['__all__']
        self._frame_count += 1
        return obs, self.done, reward

    def _convert_actions(self, pseudo_actions_tensor):
        def _convert(handle):
            act = pseudo_actions_tensor[handle]
            if act == 0:
                return 4  # STOP

            allowed_dirs = self._obs_builder.allowed_directions[handle]
            # Example of allowed dirs: ['L', 'F']
            if len(allowed_dirs) == 0:
                return 0  # DO NOTHING (keep going)
            if len(allowed_dirs) == 1:
                return 2  # FORWARD

            chosen_dir = allowed_dirs[act - 1]
            dirmap = {"L": 1, "F": 2, "R": 3}
            # print(allowed_dirs, int(act), chosen_dir)
            return dirmap.get(chosen_dir, 0)

        # Iterate through the pseudo-action tensor, building the real action dict
        return {h: _convert(h) for h in range(len(pseudo_actions_tensor))}

    def _agent_reward(self, handle):
        agent = self._env.agents[handle]
        reward = -1

        if agent.status in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED):
            # If the train has reached the target, increase the reward
            reward += 2
        else:
            # Decrease the reward by the distance (manhattan) from its target
            p, t = agent.position, agent.target
            dist_from_target = abs(p[0] - t[0]) + abs(p[1] - t[1])
            reward -= dist_from_target

            # print(agent.malfunction_data['malfunction'])

        return reward
