from __future__ import division
from __future__ import print_function
import warnings
import timeit
import json
from tempfile import mkdtemp

import numpy as np

from keras import __version__ as KERAS_VERSION
from keras.callbacks import CallbackList as KerasCallbackList
from keras.utils.generic_utils import Progbar
from tensorflow.keras.callbacks import Callback as KerasCallback

###


class Callback(KerasCallback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        pass

    def on_step_begin(self, step, logs={}):
        """Called at beginning of each step"""
        pass

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, action, logs={}):
        """Called at beginning of each action"""
        pass

    def on_action_end(self, action, logs={}):
        """Called at end of each action"""
        pass


###


class CallbackList(KerasCallbackList):
    def _set_env(self, env):
        """ Set environment for each callback in callbackList """
        for callback in self.callbacks:
            if callable(getattr(callback, '_set_env', None)):
                callback._set_env(env)

    def on_episode_begin(self, episode, logs={}):
        """ Called at beginning of each episode for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_begin` callback.
            # If not, fall back to `on_epoch_begin` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_begin', None)):
                callback.on_episode_begin(episode, logs=logs)
            else:
                callback.on_epoch_begin(episode, logs=logs)

    def on_episode_end(self, episode, logs={}):
        """ Called at end of each episode for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_end` callback.
            # If not, fall back to `on_epoch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_end', None)):
                callback.on_episode_end(episode, logs=logs)
            else:
                callback.on_epoch_end(episode, logs=logs)

    def on_step_begin(self, step, logs={}):
        """ Called at beginning of each step for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_step_begin` callback.
            # If not, fall back to `on_batch_begin` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_step_begin(step, logs=logs)
            else:
                callback.on_batch_begin(step, logs=logs)

    def on_step_end(self, step, logs={}):
        """ Called at end of each step for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_step_end` callback.
            # If not, fall back to `on_batch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_step_end', None)):
                callback.on_step_end(step, logs=logs)
            else:
                callback.on_batch_end(step, logs=logs)

    def on_action_begin(self, action, logs={}):
        """ Called at beginning of each action for each callback in callbackList"""
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_begin', None)):
                callback.on_action_begin(action, logs=logs)

    def on_action_end(self, action, logs={}):
        """ Called at end of each action for each callback in callbackList"""
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_end', None)):
                callback.on_action_end(action, logs=logs)


###


class TestLogger(Callback):
    """ Logger Class for Test """
    def on_train_begin(self, logs):
        """ Print logs at beginning of training"""
        print('Testing for {} episodes ...'.format(self.params['nb_episodes']))

    def on_episode_end(self, episode, logs):
        """ Print logs at end of each episode """
        template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
        variables = [
            episode + 1,
            logs['episode_reward'],
            logs['nb_steps'],
        ]
        print(template.format(*variables))


###


class TrainEpisodeLogger(Callback):
    def __init__(self):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode (in our case maybe better to index by agent)
        # to separate episodes (or agents) from each other.
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
        self.episode_start = timeit.default_timer()
        print('\nepisode start:', self.episode_start, '\n')

    ### TODO: [@matteo -> @davide] I've renamed the {agent} param to {episode}. Ok?
    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """
        assert 'agent' in logs
        agent = logs['agent']
        duration = timeit.default_timer() - self.episode_start
        episode_steps = len(self.observations[agent])

        # Format all metrics.
        metrics = np.array(self.metrics[agent])
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
        template = 'agent: {agent}, {step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
        variables = {
            'agent': agent,
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[agent]),
            'reward_mean': np.mean(self.rewards[agent]),
            'reward_min': np.min(self.rewards[agent]),
            'reward_max': np.max(self.rewards[agent]),
            'action_mean': np.mean(self.actions[agent]),
            'action_min': np.min(self.actions[agent]),
            'action_max': np.max(self.actions[agent]),
            'obs_mean': np.mean(self.observations[agent]),
            'obs_min': np.min(self.observations[agent]),
            'obs_max': np.max(self.observations[agent]),
            'metrics': metrics_text,
        }
        print(template.format(**variables))
        """
        # Free up resources.
        # del self.episode_start[agent]
        del self.observations[agent]
        del self.rewards[agent]
        del self.actions[agent]
        del self.metrics[agent]
        """
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        assert 'agent' in logs
        agent = logs['agent']
        self.observations[agent] = self.observations.get(agent, []) + [logs['observation']]
        self.rewards[agent] = self.rewards.get(agent, []) + [logs['reward']]
        self.actions[agent] = self.actions.get(agent, []) + [logs['action']]
        self.metrics[agent] = self.metrics.get(agent, []) + [logs['metrics']]
        self.step += 1


###


class TrainIntervalLogger(Callback):
    def __init__(self, interval=10000):
        self.interval = interval
        self.step = 0
        self.reset()

    def reset(self):
        """ Reset statistics """
        self.interval_start = timeit.default_timer()
        self.progbar = Progbar(target=self.interval)
        self.metrics = []
        self.infos = {}
        self.info_names = None
        self.episode_rewards = {}

    def set_model(self, model):
        self.model = model
        self.metrics_names = self.model.metrics_names

    def on_train_begin(self, logs):
        """ Initialize training statistics at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training duration at end of training """
        duration = timeit.default_timer() - self.train_start
        print('\ndone, took {:.3f} seconds'.format(duration))

    def on_step_begin(self, step, logs):
        """ Print metrics if interval is over """
        assert 'agent' in logs
        agent = logs['agent']

        if self.step > 0 and self.step % self.interval == 0:
            ### ISSUE: [@matteo] sometimes {metrics_names} is empty ...
            ### INFO: I've tried to fix it with the following lines
            # if len(self.model.metrics_names) > 0:
            # self.metrics_names = self.model.metrics_names
            assert len(self.metrics_names) > 0

            if len(self.episode_rewards.get(agent, [])) > 0:
                metrics = np.array(self.metrics)
                assert metrics.shape == (self.interval, len(self.metrics_names))
                formatted_metrics = ''
                if not np.isnan(metrics).all():  # not all values are means
                    means = np.nanmean(self.metrics, axis=0)
                    assert means.shape == (len(self.metrics_names), )
                    for name, mean in zip(self.metrics_names, means):
                        formatted_metrics += ' - {}: {:.3f}'.format(name, mean)

                formatted_infos = ''
                if len(self.infos.get(agent, [])) > 0:
                    infos = np.array(self.infos[agent])
                    if not np.isnan(infos).all():  # not all values are means
                        means = np.nanmean(self.infos[agent], axis=0)
                        assert means.shape == (len(self.info_names), )
                        for name, mean in zip(self.info_names, means):
                            formatted_infos += ' - {}: {:.3f}'.format(name, mean)
                print(
                    '{} episodes - episode_reward: {:.3f} [{:.3f}, {:.3f}]{}{}'.format(
                        len(self.episode_rewards[agent]), np.mean(self.episode_rewards[agent]),
                        np.min(self.episode_rewards[agent]), np.max(self.episode_rewards[agent]),
                        formatted_metrics, formatted_infos
                    )
                )
                print('')
            self.reset()
            print(
                'Agent {} | Interval {} | Steps {}'.format(
                    agent, self.step // self.interval + 1, self.step
                )
            )

    def on_step_end(self, step, logs):
        """ Update progression bar at the end of each step """
        # don't know how to insert agent handle
        assert 'agent' in logs
        agent = logs['agent']
        if self.info_names is None:
            self.info_names = logs['info'].keys()
        values = [('reward', logs['reward'])]

        # don't know why, but self.progbar does not display each time the function is called
        if KERAS_VERSION > '2.1.3':
            self.progbar.update((self.step % self.interval) + 1, values=values)
        else:
            self.progbar.update((self.step % self.interval) + 1, values=values, force=True)

        # update step and append metrics only once
        if agent == 0:
            self.step += 1
            self.metrics.append(logs['metrics'])

        if len(self.info_names) > 0:
            self.infos[agent] = self.infos.get(agent,
                                               []) + [[logs['info'][k] for k in self.info_names]]

    def on_episode_end(self, episode, logs):
        """ Update reward value at the end of each episode """
        assert 'agent' in logs
        agent = logs['agent']
        self.episode_rewards[agent] = self.episode_rewards.get(agent, []) + [logs['episode_reward']]


###


class FileLogger(Callback):
    def __init__(self, filepath, interval=None):
        self.filepath = filepath
        self.interval = interval

        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        self.metrics = {}
        self.starts = {}
        self.data = {}

    def on_train_begin(self, logs):
        """ Initialize model metrics before training """
        self.metrics_names = self.model.metrics_names

    def on_train_end(self, logs):
        """ Save model at the end of training """
        self.save_data()

    def on_episode_begin(self, episode, logs):
        """ Initialize metrics at the beginning of each episode """
        assert episode not in self.metrics
        assert episode not in self.starts
        self.metrics[episode] = []
        self.starts[episode] = timeit.default_timer()

    def on_episode_end(self, episode, logs):
        """ Compute and print metrics at the end of each episode """
        duration = timeit.default_timer() - self.starts[episode]

        metrics = self.metrics[episode]
        if np.isnan(metrics).all():
            mean_metrics = np.array([np.nan for _ in self.metrics_names])
        else:
            mean_metrics = np.nanmean(metrics, axis=0)
        assert len(mean_metrics) == len(self.metrics_names)

        data = list(zip(self.metrics_names, mean_metrics))
        data += list(logs.items())
        data += [('episode', episode), ('duration', duration)]
        for key, value in data:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

        if self.interval is not None and episode % self.interval == 0:
            self.save_data()

        # Clean up.
        del self.metrics[episode]
        del self.starts[episode]

    def on_step_end(self, step, logs):
        """ Append metric at the end of each step """
        self.metrics[logs['episode']].append(logs['metrics'])

    def save_data(self):
        """ Save metrics in a json file """
        if len(self.data.keys()) == 0:
            return

        # Sort everything by episode.
        assert 'episode' in self.data
        sorted_indexes = np.argsort(self.data['episode'])
        sorted_data = {}
        for key, values in self.data.items():
            assert len(self.data[key]) == len(sorted_indexes)
            # We convert to np.array() and then to list to convert from np datatypes to native datatypes.
            # This is necessary because json.dump cannot handle np.float32, for example.
            sorted_data[key] = np.array([self.data[key][idx] for idx in sorted_indexes]).tolist()

        # Overwrite already open file. We can simply seek to the beginning since the file will
        # grow strictly monotonously.
        with open(self.filepath, 'w') as f:
            json.dump(sorted_data, f)


###


class Visualizer(Callback):
    def on_action_end(self, action, logs):
        """ Render environment at the end of each action """
        self.env.render(mode='human')


###


class ModelIntervalCheckpoint(Callback):
    def __init__(self, filepath, interval, verbose=0):
        super(ModelIntervalCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)

        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))

        self.model.save_weights(filepath, overwrite=True)
