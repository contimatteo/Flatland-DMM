from datetime import datetime

from tensorforce import Environment, Agent
from tensorforce.execution import Runner

from msrc import config
from msrc.environment import TFEnvironment
# Environment
from msrc.network import get_model

env = Environment.create(
    environment=TFEnvironment(render=False), max_episode_timesteps=config.ENV_MAX_TIMESTEPS
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
    # saver=dict(
    #     directory='model-checkpoint',
    #     frequency=config.SAVE_FREQUENCY  # save checkpoint every 100 updates
    # ),
)
# agent = Agent.load(directory='model-checkpoint', format='checkpoint', environment=environment)

# Main loop (training)
for episode in range(config.TRAINING_EPISODES):
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
agent.save(directory='model-checkpoint', filename=datetime.now().strftime("%Y%m%d %H%M%S"))

# Evaluation loop
env = Environment.create(
    environment=TFEnvironment(render=True), max_episode_timesteps=config.ENV_MAX_TIMESTEPS
)
for _ in range(config.EVALUATION_EPISODES):
    obs = env.reset()
    done = False
    tot_reward = 0

    while not done:
        actions = agent.act(states=obs)
        obs, done, reward = env.execute(actions)
        tot_reward += reward
    print(":::TESTING::: Done EP", episode, "with reward", tot_reward)