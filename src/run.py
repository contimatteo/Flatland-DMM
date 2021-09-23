import os
import argparse
import sys
from pathlib import Path

from configs import configurator as Configurator
from core import Runner

###


def run(config_filepath: str = None):
    configs = []

    if config_filepath is None:
        config_filepath = './src/configs/run.json'
    
    root_dir = Path(os.path.abspath(__file__)).parent.parent
    default_config_filepath = root_dir.joinpath(config_filepath)
    assert default_config_filepath.is_file() is True
    configs = Configurator.get_configs_from_file(str(default_config_filepath.absolute()))

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


def parse_args():
    # Create the parser
    my_parser = argparse.ArgumentParser(
        prog='Flatland-DMM', usage='%(prog)s [options] config-file', description='TODO: ...'
    )

    # Add the arguments
    my_parser.add_argument(
        '--config',
        type=str,
        required=False,
        help='path of the json file with the running configurations.',
    )

    # Execute the parse_args() method
    args = my_parser.parse_args()

    # config_file_path = str(Path(__file__).parent.parent.joinpath(args.config).absolute())
    config_file_path = args.config

    if config_file_path is not None and not os.path.isfile(config_file_path):
        print('The file specified does not exist')
        sys.exit()

    return config_file_path


###

if __name__ == '__main__':
    config_file_path = parse_args()
    run(config_file_path)
