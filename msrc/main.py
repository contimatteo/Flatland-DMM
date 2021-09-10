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
    memory=100 + 20,
    network=dict(type='keras', model=model),
)

# Main loop (training)
for episode in range(config.TRAINING_EPISODES):
    obs = env.reset()
    done = False
    print("Start EP", episode)
    while not done:
        actions = agent.act(states=obs)
        obs, done, reward = env.execute(actions)
        agent.observe(terminal=done, reward=reward)

# Evaluation loop
runner = Runner(
    agent=agent,
    environment=Environment.create(
        environment=TFEnvironment(render=True), max_episode_timesteps=config.ENV_MAX_TIMESTEPS
    ),
    max_episode_timesteps=config.ENV_MAX_TIMESTEPS
)
runner.run(num_episodes=config.EVAL_EPISODES)
