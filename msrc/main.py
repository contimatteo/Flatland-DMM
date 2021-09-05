from tf_agents.environments import validate_py_environment

from environment import FLEnvironment
from msrc import config

# Environment
from msrc.network import ActorNetwork

env = FLEnvironment(render=True)
# validate_py_environment(env, episodes=5)  # PASSES

# Agent (single)
actor = ActorNetwork(env.observation_spec())

# Main loop
for episode in range(config.ENV_MAX_EPISODES):
    time_step = env.reset()
    while not env.done:
        action, _ = actor(time_step.observation, time_step.step_type)
        time_step = env.step(action)
