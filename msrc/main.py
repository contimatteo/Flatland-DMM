from tf_agents.environments import validate_py_environment

from environment import FLEnvironment
from msrc import config
from msrc.agent import SimpleAgent

# Environment
env = FLEnvironment(render=True)
validate_py_environment(env, episodes=5)  # PASSES

# Agent (single)
agent = SimpleAgent(env)

# Main loop
for episode in range(config.ENV_MAX_EPISODES):
    time_step = env.reset()

    while not env.done:
        action = agent.act(time_step)
        time_step = env.step(action)
