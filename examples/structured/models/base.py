###


class BaseModel:
    def __init__(self, name='BaseModel'):
        self.name = name

    def initialize(self):
        raise Exception('not implemented.')

    def train(self):
        raise Exception('not implemented.')

    ###

    @staticmethod
    def compile_model(input_nodes, input_dim, output_nodes):
        raise Exception('not implemented.')
