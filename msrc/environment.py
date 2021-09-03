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
            self.renderer = RenderTool(self._env, show_debug=False, screen_height=1000, screen_width=1000)

        # SPECS
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(config.N_AGENTS,), dtype=np.int32, minimum=0, maximum=4, name='action')

        # OTHER
        self.info = None
        self.done = False
        self._frame_count = 0

    def get_fl_env(self):
        return self._env

    def observation_spec(self) -> types.NestedArraySpec:
        return self._obs_builder.obs_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def get_info(self) -> types.NestedArray:
        pass

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        # End of episode (all done or frames limit)
        if self.done or self._frame_count > config.ENV_MAX_FRAMES:
            return self._reset()

        # Convert agent's action vector into dictionary, then pass it to the fl environment
        action_dict = {h: action[h] for h in range(len(action))}
        obs, reward_dict, done_dict, self.info = self._env.step(action_dict)

        # Reward determination TODO: weight train collisions and other weighting
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
