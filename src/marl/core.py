import warnings
from copy import deepcopy

import numpy as np
from tensorflow.keras.callbacks import History

from rl.core import Agent
from rl.callbacks import (
    CallbackList, TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer
)

###


class MultiAgent(Agent):
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

        ### VALIDATION

        ### {env}
        assert env is not None
        ### {nb_steps}
        assert isinstance(nb_steps, int) or isinstance(nb_steps, float)
        assert nb_steps > 0
        ### {verbose}
        assert verbose is not None
        assert isinstance(verbose, int)

        if not self.compiled:
            error_msg = 'Your tried to fit your agent but it hasn\'t been compiled yet.'
            raise RuntimeError(error_msg)

        ### CALLBACKS

        callbacks_funcs = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks_funcs += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks_funcs += [TrainEpisodeLogger()]
        if visualize:
            callbacks_funcs += [Visualizer()]

        history = History()
        callbacks_funcs += [history]

        ### create the callbacks handler
        callbacks = CallbackList(callbacks=callbacks_funcs)

        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        # else: callbacks._set_model(self)

        callbacks._set_env(env)  # TODO: check this ...

        params = {'nb_steps': nb_steps}
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        # else: callbacks._set_params(params)

        ###

        self._on_train_begin()
        callbacks.on_train_begin()

        did_abort = False
        episode_step = None
        episode = np.int16(0)
        self.step = np.int16(0)
        observations_dict = None
        episode_rewards_dict = None

        try:
            while self.step < nb_steps:
                if observations_dict is None:
                    ### start of a new episode
                    episode_rewards_dict = {}
                    episode_step = np.int16(0)

                    ### callback (call)
                    callbacks.on_episode_begin(episode)

                    self.reset_states()

                    ### Obtain the initial observation by resetting the environment.
                    observations_dict = env.reset()

                    observations_dict = deepcopy(observations_dict)
                    if self.processor is not None:
                        observations_dict = self.processor.process_observation(observations_dict)

                    assert observations_dict is not None

                ### At this point, we expect to be fully initialized.
                assert episode_step is not None
                assert observations_dict is not None
                assert episode_rewards_dict is not None

                all_done = False
                actions_dict = {}
                n_agents = len(observations_dict)

                ### GET ACTIONS

                for agent_id in range(n_agents):
                    ### callback (call)
                    callbacks.on_step_begin(episode_step, logs={'agent_id': agent_id})

                    ### This is were all of the work happens. We first perceive and compute the
                    ### action (forward step) and then use the reward to improve (backward step).
                    act = self.forward(observations_dict.get(agent_id))

                    if self.processor is not None:
                        act = self.processor.process_action(act)
                    actions_dict[agent_id] = act

                    ### callback (call)
                    callbacks.on_action_begin(action=act)

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

                accumulated_info = {}

                # for agent_id in range(n_agents):
                # for key, value in info_dict.items():
                #     if not np.isreal(value):
                #         continue
                #     if key not in accumulated_info:
                #         accumulated_info[key] = np.zeros_like(value)
                #     accumulated_info[key] += value

                ### COLLECT REWARDS

                for agent_id in range(n_agents):
                    episode_rewards_dict[
                        agent_id] = episode_rewards_dict.get(agent_id, 0) + rewards_dict[agent_id]

                ### STEP TERMINATION CONDITIONS

                for agent_id in range(n_agents):
                    ### callback (call)
                    callbacks.on_action_end(actions_dict.get(agent_id))

                done_dict_values_as_sum = 0
                for agent_id in range(n_agents):
                    done_dict_values_as_sum += int(done_dict.get(agent_id, False))
                if done_dict_values_as_sum == n_agents:
                    all_done = True

                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    ### force a terminal state.
                    all_done = True

                ### TRAINING

                ### TODO: reason about calling this for each agent.
                metrics = self.backward(rewards_dict, terminal=done_dict)

                ### METRICS

                for agent_id in range(n_agents):
                    step_logs = {
                        'agent_id': agent_id,
                        'action': actions_dict[agent_id],
                        'observation': observations_dict[agent_id],
                        'reward': rewards_dict[agent_id],
                        'metrics': metrics,
                        'episode': episode,
                        'info': accumulated_info,
                    }
                    ### callback (call)
                    callbacks.on_step_end(episode_step, step_logs)

                episode_step += 1
                self.step += 1

                if all_done is True:
                    ### We are in a terminal state but the agent hasn't yet seen it. We therefore
                    ### perform one more forward-backward call and simply ignore the action before
                    ### resetting the environment. We need to pass in `terminal=False` here since
                    ### the *next* state, that is the state of the newly reset environment, is
                    ### always non-terminal by convention.

                    for agent_id in range(n_agents):
                        self.forward(observations_dict.get(agent_id))

                        ### This episode is finished, report and reset.
                        episode_logs = {
                            'agent_id': agent_id,
                            'episode_reward': episode_rewards_dict[agent_id],
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                        }

                        ### callback (call)
                        callbacks.on_episode_end(episode, episode_logs)

                    ### TODO: reason about calling this for each agent.
                    ### TODO: ask to @davide why he have put {False} in the {terminal} parameter.
                    self.backward(0., terminal=False)

                    episode += 1
                    observations_dict = None
                    episode_step = None
                    episode_rewards_dict = None

        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True

        ### callback (call)
        callbacks.on_train_end(logs={'did_abort': did_abort})

        self._on_train_end()

        return history

    def test(
        self,
        env,
        nb_episodes=1,
        action_repetition=1,
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

        ### VALIDATION

        ### {env}
        assert env is not None
        ### {verbose}
        assert verbose is not None
        assert isinstance(verbose, int)

        if not self.compiled:
            error_msg = 'Your tried to fit your agent but it hasn\'t been compiled yet.'
            raise RuntimeError(error_msg)

        ### CALLBACKS

        callbacks_funcs = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks_funcs += [TestLogger()]
        if visualize:
            callbacks_funcs += [Visualizer()]

        history = History()
        callbacks_funcs += [history]

        ### create the callbacks handler
        callbacks = CallbackList(callbacks_funcs)

        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        # else: callbacks._set_model(self)

        callbacks._set_env(env)  # TODO: check this ...

        params = {'nb_episodes': nb_episodes}
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        # else: callbacks._set_params(params)

        ###

        self.step = 0

        self._on_test_begin()

        ### callback (call)
        callbacks.on_train_begin()

        for episode in range(nb_episodes):
            episode_step = 0
            actions_dict = {}
            episode_rewards_dict = {}

            ### callback (call)
            callbacks.on_episode_begin(episode)

            self.reset_states()

            ### obtain the initial observation by resetting the environment.
            observations_dict = env.reset()

            observations_dict = deepcopy(observations_dict)
            if self.processor is not None:
                observations_dict = self.processor.process_observation(observations_dict)

            assert observations_dict is not None

            all_done = False

            ### run the episode until we're done.
            while not all_done:
                n_agents = len(observations_dict)

                ### GET ACTIONS

                for agent_id in range(n_agents):
                    ### callback (call)
                    callbacks.on_step_begin(episode_step, logs={'agent_id': agent_id})

                    act = self.forward(observations_dict.get(agent_id))

                    if self.processor is not None:
                        act = self.processor.process_action(act)
                    actions_dict[agent_id] = act

                    ### callback (call)
                    callbacks.on_action_begin(actions_dict.get(agent_id))

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

                accumulated_info = {}

                # for agent_id in range(n_agents):
                # for key, value in info_dict.items():
                #     if not np.isreal(value):
                #         continue
                #     if key not in accumulated_info:
                #         accumulated_info[key] = np.zeros_like(value)
                #     accumulated_info[key] += value

                ### COLLECT REWARDS

                for agent_id in range(n_agents):
                    episode_rewards_dict[
                        agent_id] = episode_rewards_dict.get(agent_id, 0) + rewards_dict[agent_id]

                ###

                for agent_id in range(n_agents):
                    ### callback (call)
                    callbacks.on_action_end(actions_dict.get(agent_id))

                ### TESTING

                ### TODO: reason about calling this for each agent.
                self.backward(rewards_dict, terminal=done_dict)

                for agent_id in range(n_agents):
                    step_logs = {
                        'agent_id': agent_id,
                        'action': actions_dict[agent_id],
                        'observation': observations_dict[agent_id],
                        'reward': rewards_dict[agent_id],
                        'episode': episode,
                        'info': accumulated_info,
                    }
                    callbacks.on_step_end(episode_step, step_logs)

                episode_step += 1
                self.step += 1

            ### We are in a terminal state but the agent hasn't yet seen it. We therefore
            ### perform one more forward-backward call and simply ignore the action before
            ### resetting the environment. We need to pass in `terminal=False` here since
            ### the *next* state, that is the state of the newly reset environment, is
            ### always non-terminal by convention.
            for agent_id in range(n_agents):
                self.forward(observations_dict.get(agent_id))

                ### report end of episode.
                episode_logs = {
                    'agent_id': agent_id,
                    'episode_reward': episode_rewards_dict[agent_id],
                    'nb_steps': episode_step,
                }

                ### callback (call)
                callbacks.on_episode_end(episode, episode_logs)

            ### TODO: reason about calling this for each agent.
            ### TODO: ask to @davide why he have put {False} in the {terminal} parameter.
            self.backward(0., terminal=False)

        ### callback (call)
        callbacks.on_train_end()

        self._on_test_end()

        return history
