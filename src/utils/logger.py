import logging
import logging.config

from pathlib import Path

###

ROOT_DIR_PATH = Path(__file__).parent.parent
FILE_DIR_PATH = "tmp/logs"
FILE_NAME = "out.log"

FILE_PATH = ROOT_DIR_PATH.joinpath(FILE_DIR_PATH)
FILE_URI = FILE_PATH.joinpath(FILE_NAME)

###

console = None
file = None

logging.getLogger('tensorflow').disabled = True

FILE_PATH.mkdir(parents=True, exist_ok=True)

###

# https://stackoverflow.com/a/56144390
logging.root.setLevel(logging.NOTSET)

# Create a custom logger
console = logging.getLogger('console')
file = logging.getLogger('file')

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(filename=FILE_URI)
# c_handler.setLevel(logging.WARNING)
# f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter('[%(levelname)s]  %(message)s')
f_format = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
console.addHandler(c_handler)
file.addHandler(f_handler)
