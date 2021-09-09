from keras.layers import Dense
from keras import Sequential


import configs as Configs

def build_model(input_dim, action_size):

    model = Sequential()

    model.add(Dense(32, input_dim=input_dim, activation=Configs.ACTIVATION_HIDDEN_L))
    model.add(Dense(16, activation=Configs.ACTIVATION_HIDDEN_L))
    model.add(Dense(action_size, activation=Configs.ACTIVATION_OUTPUT_L))

    """    
    model.compile(optimizer=Adam(lr=Configs.LEARNING_RATE),
                  loss=Configs.LOSS_FUNCTION)
    """

    return model



