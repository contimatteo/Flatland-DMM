import time
from flatland.utils.rendertools import RenderTool
from environment import FLEnvironment
from msrc import config
from msrc.agent import TreeLookupAgent

# Environment
env = FLEnvironment().get_env()
renderer = RenderTool(
    env,
    show_debug=False,
    screen_height=1000,
    screen_width=1000)

# Agent (single)
agent = TreeLookupAgent()

# Main loop
for episode in range(config.ENV_MAX_EPISODES):
    obs, info = env.reset(activate_agents=True)
    renderer.reset()

    for frame in range(config.ENV_MAX_FRAMES):
        actions = {handle: agent.act(handle, obs, info) for handle in range(config.N_AGENTS)}
        obs, reward, done, info = env.step(actions)
        renderer.render_env(show=True, show_observations=False)
        time.sleep(0.01)
        if done['__all__']:
            break
