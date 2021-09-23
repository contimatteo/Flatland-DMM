from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

from networks.base import BaseNetwork

###

LEARNING_RATE = 0.01

###


class SequentialNetwork1(BaseNetwork):
    @property
    def uuid(self) -> str:
        return 'conv-1'

    ###

    def build_model(self) -> Sequential:
        model = Sequential()

        model.add(self.input_layer())

        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))

        model.add(self.output_layer())

        print(model.summary())

        return model
