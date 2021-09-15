from tensorforce import Environment, Agent

from msrc import config
from msrc.environment import TFEnvironment
from msrc.network import get_model
from msrc.runner import FLRunner

IS_TRAINING = True
MODEL_FILEPATH = "model-checkpoint/2021 09 15 A.h5"

env = Environment.create(
    environment=TFEnvironment(render=False, regeneration_frequency=config.ENV_REGENERATION_FREQUENCY),
    max_episode_timesteps=config.ENV_MAX_TIMESTEPS
)

# Agent (single)
model = get_model(env.states()['shape'])
model.summary()
if not IS_TRAINING:
    model.load_weights(MODEL_FILEPATH, by_name=True)

agent = Agent.create(
    agent='dqn',
    environment=env,
    batch_size=100,
    horizon=1,
    memory=config.ENV_MAX_TIMESTEPS + 20,
    network=dict(type='keras', model=model),
    exploration=0.1 * IS_TRAINING
)

if IS_TRAINING:
    FLRunner.fit(agent, env, model)
else:
    FLRunner.eval(agent, env)
