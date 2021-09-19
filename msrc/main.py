from tensorforce import Environment

from msrc import config
from msrc.agent import load_agent, create_agent
from msrc.config import IS_TRAINING, PRELOAD_MODEL
from msrc.environment import TFEnvironment
from msrc.runner import FLRunner

env = Environment.create(
    environment=TFEnvironment(
        render=not IS_TRAINING,
        regeneration_frequency=config.ENV_REGENERATION_FREQUENCY * IS_TRAINING),
    max_episode_timesteps=config.ENV_MAX_TIMESTEPS
)

agent_type, network_type = 'dqn', 'conv1d_to_dense'
save_name = agent_type + '-' + network_type

# Agent (single)
if IS_TRAINING and not PRELOAD_MODEL:
    agent = create_agent(agent_type, network_type)
else:
    agent = load_agent(agent_type, save_name)

print(agent.get_architecture())
print(agent.get_specification())
print(agent.tracked_tensors())
print()
print()

# Runner
if IS_TRAINING:
    FLRunner.fit(agent, env, save_name=save_name)
else:
    FLRunner.eval(agent, env)
