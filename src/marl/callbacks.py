from __future__ import division
from __future__ import print_function
import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np
import wandb

from keras import __version__ as KERAS_VERSION
from keras.callbacks import CallbackList as KerasCallbackList
from keras.utils.generic_utils import Progbar
from tensorflow.keras.callbacks import Callback as KerasCallback

###


class WandbLogger(KerasCallback):
    """ Similar to TrainEpisodeLogger, but sends data to Weights & Biases to be visualized """
    def __init__(self, **kwargs):
        kwargs = {'anonymous': 'allow', **kwargs}
        wandb.init(reinit=True, **kwargs)
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0

    def _set_env(self, env):
        self.env = env

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        wandb.config.update(
            {
                'params': self.params,
                'env': self.env.__dict__,
                'agent': self.model.__dict__,
                'env.env': self.env._env.__dict__
                # 'env.env.spec': self.env._env.spec.__dict__,
            },
            allow_val_change=True
        )

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and log training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        metrics = np.array(self.metrics[episode])
        metrics_dict = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    metrics_dict[name] = np.nanmean(metrics[:, idx])
                except Warning:
                    metrics_dict[name] = float('nan')

        wandb.log(
            {
                'step': self.step,
                'episode': episode + 1,
                # 'duration': duration,
                'episode_steps': episode_steps,
                # 'sps': float(episode_steps) / duration,
                'episode_reward_sum': np.sum(self.rewards[episode]),
                'reward_mean': np.mean(self.rewards[episode]),
                'reward_min': np.min(self.rewards[episode]),
                'reward_max': np.max(self.rewards[episode]),
                # 'action_mean': np.mean(self.actions[episode]),
                # 'action_min': np.min(self.actions[episode]),
                # 'action_max': np.max(self.actions[episode]),
                # 'obs_mean': np.mean(self.observations[episode]),
                # 'obs_min': np.min(self.observations[episode]),
                # 'obs_max': np.max(self.observations[episode]),
                'steps': logs['steps'],
                'target_reached': logs['target_reached'],
                'target_reached_in_steps': logs['target_reached_in_steps'],
                'episode_reward': logs['episode_reward'],
                **metrics_dict
            }
        )

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1


###


class TrainEpisodeLogger(KerasCallback):
    def __init__(self):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0

    def on_train_begin(self, logs):
        """ Print training values at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        # template = '{step: nb_step_digits}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
        template = '(( episode: {episode}, step: {steps}/{nb_steps}, episode_steps: {episode_steps}   ---   target_reached: {target_reached}, target_reached_in_steps: {target_reached_in_steps}   ---   episode_reward: {episode_reward:.3f}, mean_reward: {reward_mean:.3f}   ---   metrics: {metrics} ))'
        variables = {
            # 'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            # 'duration': duration,
            'episode_steps': episode_steps,
            # 'sps': float(episode_steps) / duration,
            # 'episode_reward_sum': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            # 'reward_min': np.min(self.rewards[episode]),
            # 'reward_max': np.max(self.rewards[episode]),
            # 'action_mean': np.mean(self.actions[episode]),
            # 'action_min': np.min(self.actions[episode]),
            # 'action_max': np.max(self.actions[episode]),
            # 'obs_mean': np.mean(self.observations[episode]),
            # 'obs_min': np.min(self.observations[episode]),
            # 'obs_max': np.max(self.observations[episode]),
            'steps': logs['steps'],
            'target_reached': logs['target_reached'],
            'target_reached_in_steps': logs['target_reached_in_steps'],
            'episode_reward': logs['episode_reward'],
            'metrics': metrics_text,
        }
        print(template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1


###

class TestLogger(KerasCallback):
    """ Logger Class for Test """
    def on_train_begin(self, logs):
        """ Print logs at beginning of training"""
        print(f"Testing for {self.params['nb_episodes']} episodes ...")

    def on_episode_end(self, episode, logs):
        """ Print logs at end of each episode """
        template = '(( episode {0}: steps: {1}   ---   target_reached: {2}, target_reached_in_steps: {3}   ---   reward: {4:.3f} ))'
        variables = [
            episode + 1,
            logs['nb_steps'],
            logs['target_reached'],
            logs['target_reached_in_steps'],
            logs['episode_reward'],
        ]
        print(template.format(*variables))
