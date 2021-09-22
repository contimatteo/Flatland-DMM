import os
from pathlib import Path

from configs import configurator as Configurator
from core import Runner

###


def run(config_filepath: str = None):
    configs = []

    if config_filepath is None:
        root_dir = Path(os.path.abspath(__file__)).parent
        default_config_filepath = str(root_dir.joinpath('configs/run.test.json').absolute())
        configs = Configurator.get_configs_from_file(default_config_filepath)

    if configs is None or len(configs) == 0:
        Configurator.reset()

        runner = Runner()

        if 'train' in configs:
            runner.train()
        if 'test' in configs:
            runner.test()
    else:
        Configurator.reset()
        
        for config in configs:
            Configurator.load_configs(config)

            runner = Runner()

            if 'train' in config:
                runner.train()
            if 'test' in config:
                runner.test()


###

if __name__ == '__main__':
    run()
