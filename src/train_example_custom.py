import time

from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
import flatland.envs.rail_generators as rail_gen
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from src.random_agent import RandomAgent


# RAIL GENERATION
print("Generating rail...")
rail_generator = rail_gen.sparse_rail_generator(
    max_num_cities=5,
    seed=123,
    grid_mode=True,
    max_rails_between_cities=2,
    max_rails_in_city=2
)

speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

# We can now initiate the schedule generator with the given speed profiles

schedule_generator = sparse_schedule_generator(speed_ration_map)
n_agents = 1
tree_max_depth = 3
rail_environment = RailEnv(
    width=20,
    height=20,
    rail_generator= rail_generator,
    schedule_generator=schedule_generator,
    number_of_agents=n_agents,
    obs_builder_object=TreeObsForRailEnv(max_depth=tree_max_depth, predictor=ShortestPathPredictorForRailEnv()),
    remove_agents_at_target=True,
    record_steps=True
)
obs, info = rail_environment.reset()

# RENDERER
print("Creating renderer...")
env_renderer = RenderTool(rail_environment,
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=True,
                          screen_height=1000,
                          screen_width=1000)
env_renderer.reset()


# AGENT
print("Creating agent...")
agent = RandomAgent(218, 4)
action_dict = dict()

# SIMULATION
print("Starting simulation")
env_renderer.render_env(show=True, show_observations=True, show_predictions=True)
time.sleep(1)
for step in range(100):
    print(f"Step: {step}")

    for a in range(rail_environment.get_num_agents()):
        action = agent.act(obs[a])
        action_dict.update({a: action})


    next_obs, all_rewards, done, _ = rail_environment.step(action_dict)
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

    for a in range(rail_environment.get_num_agents()):
        agent.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))

    obs = next_obs.copy()
    time.sleep(0.3)

    if done['__all__']:
        break
