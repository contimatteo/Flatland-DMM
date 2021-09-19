from msrc.observer import TreeTensorObserver


def get_layers_1():
    layers = [
        # Preprocessing
        dict(type='conv1d', size=16, window=1, activation='relu'),
        dict(type='conv1d', size=16, window=1, activation='relu'),
        dict(type='conv1d', size=8, window=1, activation='relu'),
        dict(type='conv1d', size=4, window=1, activation='relu'),

        # Reshaping
        dict(type='reshape', shape=(TreeTensorObserver.obs_n_nodes * 4,)),

        # Main net
        dict(type='dense', size=120, activation='relu'),
        dict(type='dense', size=80, activation='relu'),
        dict(type='dense', size=60, activation='relu'),
        dict(type='dense', size=30, activation='relu'),
        dict(type='dense', size=15, activation='relu'),

        # Last Q-values layer automatically added
    ]
    return layers


def get_tensorforce_network(network_name):
    switch = {'conv1d_to_dense': get_layers_1}
    fn = switch.get(network_name)
    return fn()
