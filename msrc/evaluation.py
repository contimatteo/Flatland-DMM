from tensorforce import Environment, Agent

from msrc import config
from msrc.environment import TFEnvironment
from msrc.network import get_model

MODEL_FILEPATH = 'model-checkpoint/20210914 181517.h5'

env = Environment.create(
    environment=TFEnvironment(render=True),
    max_episode_timesteps=config.ENV_MAX_TIMESTEPS
)

# Agent (single)
model = get_model(env.states()['shape'])
model.load_weights(MODEL_FILEPATH, by_name=True)
# model = load_model(MODEL_FILEPATH, by_name=True)
model.summary()

agent = Agent.create(
    agent='dqn',
    environment=env,
    batch_size=100,
    horizon=1,
    memory=config.ENV_MAX_TIMESTEPS + 20,
    network=dict(type='keras', model=model),
    exploration=0.05
)

# agent = Agent.load(
#     directory='model-checkpoint',
#     filename=MODEL_NAME,
#     format='hdf5',
#     environment=env,
#     agent='dqn',
#     batch_size=100,
#     horizon=1,
#     memory=config.ENV_MAX_TIMESTEPS + 20
# )

for episode in range(config.EVALUATION_EPISODES):
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
