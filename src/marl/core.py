from copy import deepcopy

import numpy as np

from tensorflow.keras.callbacks import History
from rl.core import Agent
from rl.callbacks import CallbackList
from rl.callbacks import TrainIntervalLogger
from rl.callbacks import TestLogger

from marl.callbacks import TrainEpisodeLogger
# from marl.callbacks import TrainIntervalLogger
from utils.action import HighLevelAction

###


class MultiAgent(Agent):
    def _reset_callbacks_history(self, agents_ids):
        for agent_id in agents_ids:
            self.callbacks_history[agent_id] = {
                "actions": [],
                "metrics": [],
                "observations": [],
                "rewards": [],
                "episode_reward": None,
                "target_reached": False,
                "target_reached_in_steps": 0,
            }

    def _run_callbacks(
        self, verbose, callbacks, env, training, episode, steps, nb_steps, nb_episodes, log_interval
    ):
        callbacks_funcs = [] if not callbacks else callbacks[:]

        # if verbose == 1:
        #     callbacks_funcs += [TrainIntervalLogger(interval=log_interval)]
        # elif verbose > 1:
        if training is True:
            callbacks_funcs += [TrainEpisodeLogger()]
        else:
            callbacks_funcs += [TestLogger()]

        history = History()
        callbacks_funcs += [history]

        ### create the callbacks handler
        callbacks = CallbackList(callbacks=callbacks_funcs)

        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params({'nb_steps': nb_steps, 'nb_episodes': nb_episodes})

        callbacks._set_env(env)
        callbacks.set_model(self)

        ###

        agents_ids = self.callbacks_history.keys()

        ###

        ### train/test
        if training is True:
            self._on_train_begin()
        else:
            self._on_test_begin()
        callbacks.on_train_begin()  # `keras-rl` lib  is doing this ...

        for agent_id in agents_ids:
            ep_actions = self.callbacks_history[agent_id]['actions']
            ep_observations = self.callbacks_history[agent_id]['observations']
            ep_rewards = self.callbacks_history[agent_id]['rewards']
            assert len(ep_observations) == len(ep_rewards)
            assert len(ep_observations) == len(ep_actions)

            episode_steps = len(ep_observations)

            episode_reward = self.callbacks_history[agent_id]['episode_reward']
            target_reached = self.callbacks_history[agent_id]['target_reached']
            target_reached_in_steps = self.callbacks_history[agent_id]['target_reached_in_steps']
            assert isinstance(episode_reward, float)
            assert isinstance(target_reached, bool)
            assert isinstance(target_reached_in_steps, int)

            ep_metrics = self.callbacks_history[agent_id]['metrics']

            ### episode
            callbacks.on_episode_begin(episode, logs={})

            for episode_step in range(episode_steps):
                observation = ep_observations[episode_step]
                action = ep_actions[episode_step]
                reward = ep_rewards[episode_step]
                metrics = ep_metrics[episode_step] if ep_metrics is not None else []

                assert isinstance(observation, (list, np.ndarray))
                assert isinstance(action, (int, np.uint, np.int32, np.int64))
                assert isinstance(reward, (float, np.float32, np.float64))
                assert isinstance(metrics, (list, np.ndarray))

                ### step
                callbacks.on_step_begin(episode_step, logs={})

                ### action
                callbacks.on_action_begin(action, logs={})
                callbacks.on_action_end(action, logs={})

                ### step
                callbacks.on_step_end(
                    episode_step, {
                        'action': action,
                        'observation': observation,
                        'reward': reward,
                        'metrics': metrics,
                        'episode': episode,
                        'info': {},
                    }
                )

            ### episode
            callbacks.on_episode_end(
                episode, {
                    'steps': steps,
                    'nb_steps': steps,
                    'target_reached': 1 if target_reached is True else 0,
                    'episode_reward': episode_reward,
                    'target_reached_in_steps': target_reached_in_steps,
                    'nb_episode_steps': episode_steps,
                }
            )

        ### train/test
        callbacks.on_train_end()  # `keras-rl` lib  is doing this ...
        if training is True:
            self._on_train_end()
        else:
            self._on_test_end()

    #

    def fit(
        self,
        env,
        nb_steps,
        callbacks=None,
        verbose=1,
        visualize=False,
        # nb_max_start_steps=0,
        # start_step_policy=None,
        log_interval=10000,
        nb_max_episode_steps=None
    ):
        """
        """
        self.training = True

        self.callbacks_history = {}

        ### VALIDATION

        ### {env}
        assert env is not None
        ### {nb_steps}
        assert isinstance(nb_steps, int) or isinstance(nb_steps, float)
        assert nb_steps > 0
        ### {verbose}
        assert verbose is not None
        assert isinstance(verbose, int)
        ### {nb_max_episode_steps}
        assert isinstance(nb_max_episode_steps, int) and nb_max_episode_steps > 0

        if not self.compiled:
            error_msg = 'Your tried to fit your agent but it hasn\'t been compiled yet.'
            raise RuntimeError(error_msg)

        ###

        did_abort = False
        episode_step = None
        episode = np.int16(0)
        self.step = np.int16(0)
        observations_dict = None
        episode_rewards_dict = None

        agents_ids = []

        try:
            while self.step < nb_steps:
                if observations_dict is None:
                    ### start of a new episode
                    episode_rewards_dict = {}
                    episode_step = np.int16(0)

                    self.reset_states()

                    ### Obtain the initial observation by resetting the environment.
                    observations_dict = env.reset()
                    agents_ids = [
                        agent_id for (agent_id, obs) in observations_dict.items() if obs is not None
                    ]

                    ### CALLBACK HISTORY
                    self._reset_callbacks_history(agents_ids)

                    observations_dict = deepcopy(observations_dict)
                    if self.processor is not None:
                        observations_dict = self.processor.process_observation(observations_dict)

                    assert observations_dict is not None

                ### At this point, we expect to be fully initialized.
                assert episode_step is not None
                assert observations_dict is not None
                assert episode_rewards_dict is not None

                agents_ids = [
                    agent_id for (agent_id, obs) in observations_dict.items() if obs is not None
                ]

                ### GET ACTIONS

                actions_dict = {}

                for agent_id in agents_ids:
                    ### This is were all of the work happens. We first perceive and compute the
                    ### action (forward step) and then use the reward to improve (backward step).
                    act = self.forward(observations_dict.get(agent_id), agent_id)

                    if self.processor is not None:
                        act = self.processor.process_action(act)
                    actions_dict[agent_id] = act

                ### APPLY ACTIONS

                info_dict = {}
                done_dict = {}
                rewards_dict = {}

                observations_dict, rewards_dict, done_dict, info_dict = env.step(actions_dict)

                observations_dict = deepcopy(observations_dict)
                if self.processor is not None:
                    observations_dict, rewards_dict, done_dict, info_dict = self.processor.process_step(
                        observations_dict, rewards_dict, done_dict, info_dict
                    )

                ### COLLECT INFO

                # accumulated_info = {}
                # for agent_id in agents_ids:
                # for key, value in info_dict.items():
                #     if not np.isreal(value):
                #         continue
                #     if key not in accumulated_info:
                #         accumulated_info[key] = np.zeros_like(value)
                #     accumulated_info[key] += value

                ### COLLECT REWARDS

                for agent_id in agents_ids:
                    episode_rewards_dict[
                        agent_id] = episode_rewards_dict.get(agent_id, 0) + rewards_dict[agent_id]

                ### STEP TERMINATION CONDITIONS

                all_done = False

                if done_dict.get('__all__', False) is True:
                    all_done = True

                # if all_done is False:
                #     done_dict_values_as_sum = 0
                #     for agent_id in agents_ids:
                #         done_dict_values_as_sum += int(done_dict.get(agent_id, False))
                #     if done_dict_values_as_sum == len(agents_ids):
                #         all_done = True

                if all_done is False:
                    if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                        ### force a terminal state.
                        all_done = True

                # if all_done is False:
                #     try:
                #         if episode_step + 1 >= env._env._rail_env._max_episode_steps:
                #             ### force a terminal state.
                #             all_done = True
                #     except Exception:
                #         pass

                ### TRAINING

                metrics = []

                if len(agents_ids) > 0:
                    metrics = self.backward(rewards_dict, terminal=deepcopy(done_dict))
                #     metrics = list(metrics)
                #     if len(metrics) < 1:
                #         metrics = [np.nan for _ in range(len(self.trainable_model_metrics) + 1)]
                # metrics = np.array(metrics)

                ### CALLBAKCS HISTORY

                for agent_id in agents_ids:
                    if observations_dict[agent_id] is None:
                        continue

                    self.callbacks_history[agent_id]['episode_reward'] = episode_rewards_dict[
                        agent_id]
                    self.callbacks_history[agent_id]['metrics'].append(metrics)

                    self.callbacks_history[agent_id]['observations'].append(
                        observations_dict[agent_id]
                    )
                    self.callbacks_history[agent_id]['actions'].append(actions_dict[agent_id])
                    self.callbacks_history[agent_id]['rewards'].append(rewards_dict[agent_id])

                    if done_dict[agent_id] is True:
                        if self.callbacks_history[agent_id]['target_reached'] is False:
                            self.callbacks_history[agent_id]['target_reached'] = True
                            self.callbacks_history[agent_id]['target_reached_in_steps'] = int(
                                episode_step
                            )

                ###

                episode_step += 1
                self.step += 1

                if all_done is True:
                    agents_ids = [
                        agent_id for (agent_id, obs) in observations_dict.items() if obs is not None
                    ]

                    for agent_id in agents_ids:
                        self.forward(observations_dict.get(agent_id), agent_id)

                    ### We are in a terminal state but the agent hasn't yet seen it. We therefore
                    ### perform one more forward-backward call and simply ignore the action before
                    ### resetting the environment. We need to pass in `terminal=False` here since
                    ### the *next* state, that is the state of the newly reset environment, is
                    ### always non-terminal by convention.
                    if len(agents_ids) > 0:
                        all_terminal = {i: False for i in agents_ids}
                        self.backward([0. for _ in agents_ids], terminal=all_terminal)

                    ### CALLBACKS HISTORY
                    self._run_callbacks(
                        verbose, callbacks, env, True, episode, self.step, nb_steps, None,
                        log_interval
                    )

                    ###

                    episode += 1
                    episode_step = None
                    observations_dict = None
                    episode_rewards_dict = None
                    self.callbacks_history = {}

        except KeyboardInterrupt:
            ### TODO: [@contimatteo] stop after twice {KeyboardInterrupt} errors
            ### ....

            ### We catch keyboard interrupts here so that training can be be safely aborted.
            ### This is so common that we've built this right into this function, which ensures that
            ### the `on_train_end` method is properly called.
            did_abort = True

        # return history
        return self.callbacks_history

    def test(
        self,
        env,
        nb_episodes=1,
        callbacks=None,
        visualize=True,
        nb_max_episode_steps=None,
        nb_max_start_steps=0,
        start_step_policy=None,
        verbose=1
    ):
        """
        """
        self.training = False

        self.callbacks_history = {}

        ### VALIDATION

        ### {env}
        assert env is not None
        ### {verbose}
        assert verbose is not None
        assert isinstance(verbose, int)

        if not self.compiled:
            error_msg = 'Your tried to fit your agent but it hasn\'t been compiled yet.'
            raise RuntimeError(error_msg)

        ###

        self.step = 0

        for episode in range(nb_episodes):
            episode_step = 0
            actions_dict = {}
            episode_rewards_dict = {}

            self.reset_states()

            ### obtain the initial observation by resetting the environment.
            observations_dict = env.reset()
            agents_ids = [
                agent_id for (agent_id, obs) in observations_dict.items() if obs is not None
            ]

            ### CALLBACK HISTORY
            self._reset_callbacks_history(agents_ids)

            observations_dict = deepcopy(observations_dict)
            if self.processor is not None:
                observations_dict = self.processor.process_observation(observations_dict)

            assert observations_dict is not None

            all_done = False

            ### run the episode until we're done.
            while not all_done:
                agents_ids = [
                    agent_id for (agent_id, obs) in observations_dict.items() if obs is not None
                ]

                actions_dict = {}

                for agent_id in agents_ids:
                    ### This is were all of the work happens. We first perceive and compute the
                    ### action (forward step) and then use the reward to improve (backward step).
                    act = self.forward(observations_dict.get(agent_id), agent_id)

                    if self.processor is not None:
                        act = self.processor.process_action(act)
                    actions_dict[agent_id] = act

                ### APPLY ACTIONS

                info_dict = {}
                done_dict = {}
                rewards_dict = {}

                observations_dict, rewards_dict, done_dict, info_dict = env.step(actions_dict)

                observations_dict = deepcopy(observations_dict)
                if self.processor is not None:
                    observations_dict, rewards_dict, done_dict, info_dict = self.processor.process_step(
                        observations_dict, rewards_dict, done_dict, info_dict
                    )

                ### COLLECT INFO

                # accumulated_info = {}
                # for agent_id in agents_ids:
                # for key, value in info_dict.items():
                #     if not np.isreal(value):
                #         continue
                #     if key not in accumulated_info:
                #         accumulated_info[key] = np.zeros_like(value)
                #     accumulated_info[key] += value

                ### COLLECT REWARDS

                for agent_id in agents_ids:
                    episode_rewards_dict[
                        agent_id] = episode_rewards_dict.get(agent_id, 0) + rewards_dict[agent_id]

                ### STEP TERMINATION CONDITIONS

                all_done = False

                if done_dict.get('__all__', False) is True:
                    all_done = True

                # if all_done is False:
                #     done_dict_values_as_sum = 0
                #     for agent_id in agents_ids:
                #         done_dict_values_as_sum += int(done_dict.get(agent_id, False))
                #     if done_dict_values_as_sum == len(agents_ids):
                #         all_done = True

                if all_done is False:
                    if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                        ### force a terminal state.
                        all_done = True

                # if all_done is False:
                #     try:
                #         if episode_step + 1 >= env._env._rail_env._max_episode_steps:
                #             ### force a terminal state.
                #             all_done = True
                #     except Exception:
                #         pass

                ### TESTING

                metrics = []

                if len(agents_ids) > 0:
                    metrics = self.backward(rewards_dict, terminal=done_dict)
                    metrics = list(metrics)
                    if len(metrics) < 1:
                        metrics = [np.nan for _ in range(len(self.trainable_model_metrics) + 1)]
                metrics = np.array(metrics)

                ### CALLBAKCS HISTORY

                for agent_id in agents_ids:
                    if observations_dict[agent_id] is None:
                        continue

                    self.callbacks_history[agent_id]['episode_reward'] = episode_rewards_dict[
                        agent_id]
                    self.callbacks_history[agent_id]['metrics'].append(metrics)

                    self.callbacks_history[agent_id]['observations'].append(
                        observations_dict[agent_id]
                    )
                    self.callbacks_history[agent_id]['actions'].append(actions_dict[agent_id])
                    self.callbacks_history[agent_id]['rewards'].append(rewards_dict[agent_id])

                    if done_dict[agent_id] is True:
                        if self.callbacks_history[agent_id]['target_reached'] is False:
                            self.callbacks_history[agent_id]['target_reached'] = True
                            self.callbacks_history[agent_id]['target_reached_in_steps'] = int(
                                episode_step
                            )

                episode_step += 1
                self.step += 1

            ###

            agents_ids = [
                agent_id for (agent_id, obs) in observations_dict.items() if obs is not None
            ]

            for agent_id in agents_ids:
                self.forward(observations_dict.get(agent_id), agent_id)

            ### We are in a terminal state but the agent hasn't yet seen it. We therefore
            ### perform one more forward-backward call and simply ignore the action before
            ### resetting the environment. We need to pass in `terminal=False` here since
            ### the *next* state, that is the state of the newly reset environment, is
            ### always non-terminal by convention.
            if len(agents_ids) > 0:
                all_terminal = {i: False for i in agents_ids}
                self.backward([0. for _ in agents_ids], terminal=all_terminal)

            ### CALLBACKS HISTORY
            self._run_callbacks(
                verbose, callbacks, env, False, episode, self.step, None, nb_episodes, None
            )

        return self.callbacks_history
