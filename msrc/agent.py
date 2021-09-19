from tensorforce import Agent

from msrc import config
from msrc.network import get_tensorforce_network
from msrc.observer import TreeTensorObserver


def agent_1_params():
    input_shape = (TreeTensorObserver.obs_n_nodes, TreeTensorObserver.obs_n_features)
    output_shape = (1,)
    params = dict(
        agent='dqn',
        states=dict(type='float', shape=input_shape, min_value=0.0, max_value=1.0),
        actions=dict(type='int', shape=output_shape, num_values=3),
        max_episode_timesteps=config.ENV_MAX_TIMESTEPS,
        batch_size=10,
        horizon=1,
        memory=(config.ENV_MAX_TIMESTEPS + 20) * (config.N_TRAINS + 1),
        tracking='all',
        exploration=dict(type='linear', unit='updates', num_steps=config.ENV_MAX_TIMESTEPS * 3 // 4,
                         initial_value=0.99, final_value=0.0),
    )
    return params


def create_agent(agent_type, network_type, save_name=None):
    switch = {'dqn': agent_1_params}
    params = switch.get(agent_type)()
    params['network'] = get_tensorforce_network(network_type)
    if save_name:
        pass  # auto-checkpoint-saver can be added here
    return Agent.create(**params)


def load_agent(agent_type, save_name):
    switch = {'dqn': agent_1_params}
    params = switch.get(agent_type)()
    params.update(dict(
        directory='model',
        filename=save_name,
        format='checkpoint',
    ))
    return Agent.load(**params)
