import abc

from pathlib import Path

from configs import configurator as Configs

###


class Storage():
    @staticmethod
    def initialize() -> None:
        Storage._weights_intervals_dir().mkdir(parents=True, exist_ok=True)

    #

    @staticmethod
    def _root_dir() -> Path:
        return Path(__file__).parent.parent

    @staticmethod
    def _tmp_dir() -> Path:
        return Storage._root_dir().joinpath('tmp')

    @staticmethod
    def _cache_dir() -> Path:
        return Storage._tmp_dir().joinpath(
            'cache/{}/agents-{}'.format(Configs.CONFIG_UUID, Configs.N_AGENTS)
        )

    @staticmethod
    def _weights_dir() -> Path:
        return Storage._cache_dir().joinpath('weights')

    @staticmethod
    def _weights_intervals_dir() -> Path:
        return Storage._weights_dir().joinpath('intervals')

    #

    @staticmethod
    def weights_folder() -> Path:
        assert Storage._weights_dir().is_dir()
        return Storage._weights_dir()

    @staticmethod
    def weights_intervals_folder() -> Path:
        assert Storage._weights_intervals_dir().is_dir()
        return Storage._weights_intervals_dir()
