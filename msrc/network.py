import tensorflow as tf
from keras.layers import Dense, Concatenate
from tf_agents.networks import network, Sequential

from msrc.observer import TreeTensorObserver


class ActorNetwork(network.Network):
    def __init__(self, input_tensor_spec=None, state_spec=(), name=None):
        super().__init__(input_tensor_spec, state_spec, name)

        # The obs preprocessor net takes in input a single row of the observation tensor,
        # with size equal to the len of the observed node params + extra data
        # and returns two numbers
        self._obs_preprocessor = Sequential(
            [
                Dense(8, activation=tf.tanh, input_dim=TreeTensorObserver.obs_np_len),
                Dense(4, activation=tf.tanh),
                Dense(2, activation=tf.tanh)
            ]
        )

        # Then for each of the preprocessors (which are equal to the number of explored nodes in the tree observer)
        # their output are concatenated and flattened, which is then used in the final net to determine
        # the best action for the train to take.
        # Since the actions are 3 (STOP, TURN, FORWARD),
        # there are 3 output nodes and the max value determines the action
        self._obs_grouper = Sequential(
            [
                Dense(TreeTensorObserver.obs_n_nodes, activation=tf.tanh),
                Dense(6, activation=tf.tanh),
                Dense(3, activation=tf.sigmoid)
            ]
        )

    def call(self, observations, step_type=(), network_state=()):
        # Iterate through the observations, of shape [N_TRAINS, ..., ...],
        # getting the action tensor (vector for the 3 action weights)
        # choosing the highest corresponding action
        global_action_tensors = []
        for single_train_obs in observations:
            single_action_tensor, _ = self.call_single_train(
                single_train_obs, step_type, network_state
            )
            global_action_tensors.append(single_action_tensor)

        # Concatenate the list of action tensors into a single tensor, returning it
        actions = tf.concat(global_action_tensors, 0)
        return actions, network_state

    def call_single_train(self, obs, step_type=(), network_state=()):
        # Call the network for a single train, since the net is designed to work for individual trains
        intermediate_results = []
        for n in range(TreeTensorObserver.obs_n_nodes):
            # Reshape the obs (to have a 2d tensor)
            o = tf.reshape(obs[n], (1, TreeTensorObserver.obs_np_len))
            # Apply preprocessing net, storing the result
            intermediate, network_state = self._obs_preprocessor(o, step_type, network_state)
            intermediate_results.append(intermediate)

        # Concatenate the intermediate results into the main net (grouper) input and process it,
        # getting the 3 action weights for the single train
        grouped_input = Concatenate(axis=1)(intermediate_results)
        action_tensor, network_state = self._obs_grouper(grouped_input, step_type, network_state)

        # Select the "best valued" action from the three and return it (note: it is a tensor)
        best_action = tf.math.argmax(action_tensor[0])

        return best_action, network_state
