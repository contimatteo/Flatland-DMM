from keras.optimizers import Adam
from agents.dqn_agent import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from model.build_model import build_model
from observators.tree import BinaryTreeObservator

import configs as Configs

def prepare_dqn_agent():

    model = build_model(input_dim=BinaryTreeObservator.obs_length,
                        action_size=Configs.ACTION_SIZE)

    agent = DQNAgent(model=model,
                     # policy=EpsGreedyQPolicy(),  # EpsGreedyQPolicy is already the default
                     enable_double_dqn=True,
                     nb_actions=Configs.ACTION_SIZE,
                     memory=SequentialMemory(limit=1000,
                                             window_length=1))


    agent.compile(Adam(lr=Configs.LEARNING_RATE))

    return agent