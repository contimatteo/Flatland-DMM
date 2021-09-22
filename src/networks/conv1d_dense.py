from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.python.keras import Sequential

from configs import configurator as Configs
from networks import BaseNetwork


class Conv1DDenseNetwork(BaseNetwork):
    @property
    def uuid(self) -> str:
        return 'conv1d_dense'

    def build_model(self, node_preprocessing_output_size=4) -> Sequential:
        model = Sequential()
        model.add(self.input_layer())

        # Reshape input to be matrix NODES x FEATURES
        model.add(Reshape((Configs.OBS_TREE_N_NODES, self._observations_shape[0] // Configs.OBS_TREE_N_NODES)))

        # Preprocess each node (using conv1d)
        model.add(Conv1D(30, 1, use_bias=True, activation='relu'))
        model.add(Conv1D(15, 1, use_bias=True, activation='relu'))
        model.add(Conv1D(10, 1, use_bias=True, activation='relu'))
        model.add(Conv1D(node_preprocessing_output_size, 1, use_bias=True, activation='relu'))

        # Flatten the convolved output
        model.add(Flatten(input_shape=(Configs.OBS_TREE_N_NODES, node_preprocessing_output_size)))

        # Main net that uses preprocessed inputs to determine Q-Values
        model.add(Dense(90, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(45, activation='relu'))
        model.add(Dense(15, activation='relu'))

        # Add the last layer and print the summary
        model.add(self.output_layer())
        print(model.summary())
        return model
