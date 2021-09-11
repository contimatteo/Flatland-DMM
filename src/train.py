from flatland.utils.rendertools import RenderTool, AgentRenderVariant
# from rl.agents import DQNAgent
import numpy as np

from environments.prepare_env import prepare_env
from observators.tree import BinaryTreeObservator
from agents.prepare_dqn_agent import prepare_dqn_agent

import configs as Configs

N_AGENTS = Configs.N_AGENTS
N_ATTEMPTS = Configs.TRAIN_N_ATTEMPTS
N_EPISODES = Configs.TRAIN_N_EPISODES

np.random.seed(Configs.RANDOM_SEED)

################# env setup
env = prepare_env()
env.reset()

################# agent setup
dqn_agent = prepare_dqn_agent()

nb_max_episode_steps = Configs.MAP_HEIGHT*Configs.MAP_WIDTH
nb_steps = nb_max_episode_steps * 2  # in order to have at least one env.reset() inside the fit

dqn_agent.fit(env=env,
                        nb_steps=nb_steps,
                        # nb_max_episode_steps=nb_max_episode_steps,
                        visualize=Configs.EMULATOR_ACTIVE)

dqn_agent.test(env=env,
               visualize=Configs.EMULATOR_ACTIVE)

