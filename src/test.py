import warnings
import numpy as np

from dotenv import load_dotenv

import configs as Configs

from core import Runner

warnings.filterwarnings('ignore')

###

load_dotenv()

np.random.seed(Configs.APP_SEED)

###


def test():
    runner = Runner()

    runner.test()


###

if __name__ == '__main__':
    test()
