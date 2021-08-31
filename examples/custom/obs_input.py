import numpy as np
import math
import tensorflow as tf

def from_obs_to_input(obs):
    # TODO maybe recursive
    input_list = []
    def add_node(node):
        input_list.append(node.get_attribute_list())
        for child in node.get_childs():
            if child != None:
                add_node(child)

    add_node(obs)
    flat = np.array(input_list).ravel()

    # substituting inf with 'special', but finite value
    flat[flat==math.inf] = -1
    state_tensor = tf.convert_to_tensor(flat)

    return tf.expand_dims(state_tensor, 0)
