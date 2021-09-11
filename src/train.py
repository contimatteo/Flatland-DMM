import warnings
import numpy as np

from dotenv import load_dotenv

import configs as Configs

from core.runner import Runner

warnings.filterwarnings('ignore')

###

load_dotenv()

np.random.seed(Configs.APP_SEED)

###


def train():
    runner = Runner()

    runner.train()


###

if __name__ == '__main__':
    train()
