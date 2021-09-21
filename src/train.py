import os
from pathlib import Path

from configs import configurator as Configurator
from core import Runner

###


def train(config_filepath: str = None):
    configs = []

    if config_filepath is None:
        root_dir = Path(os.path.abspath(__file__)).parent
        default_config_filepath = str(root_dir.joinpath('configs/run.train.json').absolute())
        configs = Configurator.get_configs_from_file(default_config_filepath)

    if configs is None or len(configs) == 0:
        Configurator.reset()
        
        runner = Runner()
        runner.train()
    else:
        for config in configs:
            Configurator.reset()
            Configurator.load_configs(config)

            runner = Runner()
            runner.train()


###

if __name__ == '__main__':
    train()
