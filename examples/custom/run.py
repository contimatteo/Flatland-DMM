import time
import numpy as np
import config as Configs

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.observations import LocalObsForRailEnv
from flatland.utils.rendertools import RenderTool

from agents.random import RandomAgent

###

np.random.seed(Configs.RANDOM_SEED)

N_ATTEMPTS = 1
N_EPISODES_FOREACH_ATTEMPT = 500
STATE_SIZE, ACTION_SIZE = (218, 5)

# Initialize the agent with the parameters corresponding to the environment and observation_builder
agent = RandomAgent(STATE_SIZE, ACTION_SIZE)

###


def create_env():
    tree_observator = TreeObsForRailEnv(max_depth=2)
    # tree_observator = GlobalObsForRailEnv()

    env = RailEnv(
        width=Configs.WINDOW_WIDTH,
        height=Configs.WINDOW_HEIGHT,
        number_of_agents=Configs.NUMBER_OF_AGENTS,
        rail_generator=random_rail_generator(),
        obs_builder_object=tree_observator,
    )

    env_renderer = RenderTool(env)

    return env, env_renderer


def main():
    env, env_renderer = create_env()

    # Empty dictionary for all agent action
    action_dict = dict()
    print("Starting Training...")

    for attempt in range(N_ATTEMPTS):
        # Reset environment and get initial observations for all agents
        observations, info = env.reset()

        # Reset environment renderer
        env_renderer.reset()

        # Here you can also further enhance the provided observation by means of normalization
        # See training navigation example in the baseline repository

        score = 0

        # Run episode
        for _ in range(N_EPISODES_FOREACH_ATTEMPT):
            # Chose an action for each agent in the environment
            for a in range(env.get_num_agents()):
                action = agent.act(observations[a])
                action_dict.update({a: action})

            # Environment step which returns the observations for all agents, their corresponding
            # reward and whether their are done
            next_obs, all_rewards, done, _ = env.step(action_dict)
            env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

            # Update replay buffer and train agent
            for _ in range(env.get_num_agents()):
                agent.step((observations[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
                score += all_rewards[a]

            observations = next_obs.copy()

            time.sleep(0.3)

            # should I stop ?
            if done['__all__']:
                break

        print('Episode Nr. {}\t Score = {}'.format(attempt + 1, score))


###

if __name__ == '__main__':
    main()
    input("Press Enter to continue...")
