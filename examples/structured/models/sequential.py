"""
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
"""

from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam
from rl.memory import SequentialMemory

from models.base import BaseModel

###

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

###


class SequentialModel(BaseModel):
    def __init__(self):
        super().__init__('Sequential')

    def initialize(self):
        return self

    def train(self):
        pass

    ###

    @staticmethod
    def compile_model(input_nodes, input_dim, output_nodes):
        model = Sequential()

        model.add(Dense(input_nodes, input_dim=input_dim, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(output_nodes))

        # model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE))
        model.compile(optimizer=Adam(lr=LEARNING_RATE))

        return model
