from datetime import datetime

from msrc import config


class FLRunner:
    @staticmethod
    def fit(agent, env, model, n_episodes=config.TRAINING_EPISODES, save_filepath=None):
        for episode in range(n_episodes):
            if env.episode_count % config.ENV_REGENERATION_FREQUENCY == 0:
                env_count = env.episode_count / config.ENV_REGENERATION_FREQUENCY
                print(f'============= START ENVIRONMENT {env_count} =============')

            obs = env.reset()
            done = False
            tot_reward = 0
            num_updates = 0

            while not done:
                actions = agent.act(states=obs)
                obs, done, reward = env.execute(actions)
                num_updates += agent.observe(terminal=done, reward=reward)
                tot_reward += reward
            print(":::TRAINING::: Done EP", episode, "updates:", num_updates, "with reward", tot_reward)

        if save_filepath:
            filename = f'model-checkpoint/{datetime.now().strftime("%Y%m%d %H%M%S")}.h5'
            model.save_weights(save_filepath)
            print('Saved model in', filename)
        agent.close()
        env.close()

    @staticmethod
    def eval(agent, env, n_episodes=config.EVALUATION_EPISODES):
        for episode in range(n_episodes):
            obs = env.reset()
            internals = agent.initial_internals()
            done = False
            tot_reward = 0

            while not done:
                actions = agent.act(
                    states=obs, internals=internals,
                    independent=True, deterministic=True
                )
                actions = actions[0]  # For some reason actions is a tuple
                obs, done, reward = env.execute(actions)
                tot_reward += reward
            print(":::TESTING::: Done EP", episode, "with reward", tot_reward)
        agent.close()
        env.close()
