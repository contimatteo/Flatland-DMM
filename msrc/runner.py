import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from tensorforce import Agent

from msrc import config
from msrc.environment import TFEnvironment


class FLRunner:
    @staticmethod
    def fit(agent: Agent, env: TFEnvironment, policy=lambda x: x, n_episodes=config.TRAINING_EPISODES,
            save_name="flatland_agent"):
        for episode in range(n_episodes):

            # Record episode experience
            # list of list of dicts with keys: 'states', 'internals', 'actions', 'terminal', 'reward'
            episode_record = [[] for _ in range(config.N_TRAINS)]

            # Reset environment and store agent's internals
            states = env.reset()  # Synonym of obs / observations
            internals = agent.initial_internals()
            terminal = False

            while not terminal:
                # Initialize empty action vector
                actions = np.ones(config.N_TRAINS, dtype=int) * 2
                for h in range(config.N_TRAINS):
                    is_at_switch = env.is_agent_at_switch(h)
                    if is_at_switch:
                        obs = states[h]
                        act, internals = agent.act(states=obs, internals=internals, independent=True,
                                                   deterministic=False)
                        action = policy(act)
                        actions[h] = action
                        episode_record[h].append({
                            'state': obs,
                            'internals': internals,
                            'action': act,
                            'reward': 0,
                            'terminal': False
                        })

                obs, terminal, _ = env.execute(actions)
                reward = env.reward

                for h in range(config.N_TRAINS):
                    status = env.info['status'][h]
                    agent_done = status in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED)
                    is_at_switch = env.is_agent_at_switch(h)
                    # if agent requires action => reached another node => reward can be measured
                    # Note: it measures the reward ONLY at the next node, not in the line between
                    if len(episode_record[h]) > 0 and (
                            is_at_switch or (agent_done and not episode_record[h][-1]['terminal'])):
                        episode_record[h][-1]['reward'] += reward[h]
                        episode_record[h][-1]['terminal'] = agent_done

            # At the end of the episode, compress the episode record into a dict of lists
            final_record = {k: [] for k in ('state', 'internals', 'action', 'reward', 'terminal')}
            for k in final_record:
                final_record[k] = [episode_record[i][j][k]
                                   for i in range(len(episode_record))
                                   for j in range(len(episode_record[i]))]

            # Ensure that the final episodes is terminal
            if len(final_record['terminal']) == 0:
                continue
            final_record['terminal'][-1] = 1

            # And pass the episode record to the agent's experience
            agent.experience(states=final_record['state'],
                             internals=final_record['internals'],
                             actions=final_record['action'],
                             terminal=final_record['terminal'],
                             reward=final_record['reward']
                             )
            agent.update()

            sum_rewards = sum(final_record['reward'])
            print(":::TRAINING::: Done EP", episode, "with reward", sum_rewards)

        if save_name:
            agent.save(directory='model', filename=save_name, format='checkpoint', append='timesteps')
            print(f'Saved model in /model/{save_name}')

        agent.close()
        env.close()

    # ================================================================================================================

    @staticmethod
    def eval(agent, env, policy=lambda x: x, n_episodes=config.EVALUATION_EPISODES):
        for episode in range(n_episodes):
            states = env.reset()
            internals = agent.initial_internals()
            done = False
            tot_reward = 0

            while not done:
                actions = np.ones(config.N_TRAINS, dtype=int) * 2
                for h in range(config.N_TRAINS):
                    is_at_switch = env.is_agent_at_switch(h)
                    if is_at_switch:
                        obs = states[h]
                        act, internals = agent.act(states=obs, internals=internals, independent=True)
                        action = policy(act)
                        actions[h] = action

                # actions = actions[0]  # For some reason actions is a tuple
                obs, done, _ = env.execute(actions)
                reward = env.reward

                for h in range(config.N_TRAINS):
                    status = env.info['status'][h]
                    agent_done = status in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED)
                    is_at_switch = env.is_agent_at_switch(h)
                    if is_at_switch or agent_done:
                        tot_reward += reward[h]

            print(":::TESTING::: Done EP", episode, "with reward", tot_reward)
        agent.close()
        env.close()
