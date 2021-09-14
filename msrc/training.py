from datetime import datetime

from tensorforce import Environment, Agent

from msrc import config
from msrc.environment import TFEnvironment
# Environment
from msrc.network import get_model

env = Environment.create(
    environment=TFEnvironment(render=False, regeneration_frequency=config.ENV_REGENERATION_FREQUENCY),
    max_episode_timesteps=config.ENV_MAX_TIMESTEPS
)

# Agent (single)
model = get_model(env.states()['shape'])
model.summary()

agent = Agent.create(
    agent='dqn',
    environment=env,
    batch_size=100,
    horizon=1,
    memory=config.ENV_MAX_TIMESTEPS + 20,
    network=dict(type='keras', model=model),
    exploration=0.1
)
# agent = Agent.load(directory='model-checkpoint', format='checkpoint', environment=environment)

# Main loop (training)
for episode in range(config.TRAINING_EPISODES):
    if env.episode_count % config.ENV_REGENERATION_FREQUENCY == 0:
        print(f'============= START ENVIRONMENT {env.episode_count / config.ENV_REGENERATION_FREQUENCY} =============')

    obs = env.reset()
    done = False
    tot_reward = 0

    while not done:
        actions = agent.act(states=obs)
        obs, done, reward = env.execute(actions)
        agent.observe(terminal=done, reward=reward)
        tot_reward += reward
    print(":::TRAINING::: Done EP", episode, "with reward", tot_reward)

# Save explicitly
filename = f'model-checkpoint/{datetime.now().strftime("%Y%m%d %H%M%S")}.h5'
model.save_weights(filename)
print('Saved model in', filename)
# agent.save(directory='model-checkpoint', filename=datetime.now().strftime("%Y%m%d %H%M%S"), format='hdf5')

agent.close()
env.close()
