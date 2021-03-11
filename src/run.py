import numpy as np
import config as Configs

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.utils.rendertools import RenderTool

from agent import RandomAgent

###

np.random.seed(Configs.RANDOM_SEED)

###


def create_env():
    return RailEnv(
        width=Configs.WINDOW_WIDTH,
        height=Configs.WINDOW_HEIGHT,
        number_of_agents=Configs.NUMBER_OF_AGENTS,
        rail_generator=random_rail_generator(),
        # obs_builder_object=TreeObsForRailEnv(max_depth=1),
    )


def main():
    env = create_env()
    env_renderer = RenderTool(env)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Initialize the agent with the parameters corresponding to the environment and observation_builder
    agent = RandomAgent(218, 5)
    n_trials = 5

    # Empty dictionary for all agent action
    action_dict = dict()
    print("Starting Training...")

    for trials in range(1, n_trials + 1):
        # Reset environment and get initial observations for all agents
        obs, info = env.reset()
        env_renderer.reset()

        # Here you can also further enhance the provided observation by means of normalization
        # See training navigation example in the baseline repository

        score = 0
        # Run episode
        for step in range(100):
            # Chose an action for each agent in the environment
            for a in range(env.get_num_agents()):
                action = agent.act(obs[a])
                action_dict.update({a: action})
            # Environment step which returns the observations for all agents, their corresponding
            # reward and whether their are done
            next_obs, all_rewards, done, _ = env.step(action_dict)
            env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                agent.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
                score += all_rewards[a]
            obs = next_obs.copy()
            if done['__all__']:
                break
        print('Episode Nr. {}\t Score = {}'.format(trials, score))


###

if __name__ == '__main__':
    main()
    input("Press Enter to continue...")
