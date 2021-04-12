import numpy as np
import math
import tensorflow as tf

def from_obs_to_input(obs):
    # TODO maybe recursive
    input_list = []
    def add_node(node, node_code=1):
        input_list.append(list(node[:-1]))
        input_list[-1].append(node_code)
        node_code *= 10
        for child in node.childs.values():
            node_code += 1
            if child != math.inf and child != -math.inf:
                add_node(child, node_code)

    add_node(obs)
    flat = np.array(input_list).ravel()

    # substituting inf with 'special', but finite value
    flat[flat==math.inf] = -1
    state_tensor = tf.convert_to_tensor(flat)
    return tf.expand_dims(state_tensor, 0)