from typing import Any

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.utils.rendertools import RenderTool
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

import config
from msrc.observer import TreeTensorObserver


class FLEnvironment(py_environment.PyEnvironment):
    def __init__(self, render=True):
        super().__init__()
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
            number_of_agents=config.N_AGENTS,
            remove_agents_at_target=True,
            obs_builder_object=self._obs_builder,
            random_seed=config.ENV_SEED,
            rail_generator=self._rail_gen
        )

        # RENDERER
        if render:
            self.renderer = RenderTool(
                self._env, show_debug=False, screen_height=1000, screen_width=1000
            )

        # ACTION SPECS
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(config.N_AGENTS,), dtype=np.int32, minimum=0, maximum=2, name='action'
        )

        # OTHER
        self.info = None
        self.done = False
        self._frame_count = 0

    def _step(self, pseudo_actions: types.NestedArray) -> ts.TimeStep:
        # End of episode (all done or frames limit)
        if self.done or self._frame_count > config.ENV_MAX_FRAMES:
            return self._reset()

        # Convert the pseudo actions <0,1,2> into the real action dictionary {h: <0,...,4>}
        # and pass it to the flatland environment
        action_dict = self._convert_actions(pseudo_actions)
        obs, reward_dict, done_dict, self.info = self._env.step(action_dict)

        # Reward determination
        # TODO: weight train collisions and other weighting (eg. distance) aka "Reward Shaping"
        reward = sum(reward_dict.values())

        # Rendering
        if self.renderer:
            self.renderer.render_env(show=True, show_observations=False)

        # End of episode determination and time step returning
        self.done = done_dict['__all__']
        self._frame_count += 1
        if self.done:
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=1.0)

    def _reset(self) -> ts.TimeStep:
        # Reset the environment and internal parameters
        obs, info = self._env.reset(activate_agents=True)
        self.info = info
        self.done = False
        self._frame_count = 0
        # Renderer
        if self.renderer:
            self.renderer.reset()
        return ts.restart(obs)

    def _convert_actions(self, pseudo_actions_tensor):
        def _convert(handle):
            act = pseudo_actions_tensor[handle]
            if act == 0:
                return 4  # STOP

            allowed_dirs = self._obs_builder.allowed_directions[handle]
            if len(allowed_dirs) == 0:
                return 0
            if len(allowed_dirs) == 1:
                return 2  # FORWARD

            chosen_dir = allowed_dirs[act - 1]
            dirmap = {"L": 1, "F": 2, "R": 3}
            # print(allowed_dirs, int(act), chosen_dir)
            return dirmap.get(chosen_dir, 0)

        # Iterate through the pseudo-action tensor, building the real action dict
        return {h: _convert(h) for h in range(len(pseudo_actions_tensor))}

    def observation_spec(self) -> types.NestedArraySpec:
        return self._obs_builder.obs_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def get_info(self):
        return self.info

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass
