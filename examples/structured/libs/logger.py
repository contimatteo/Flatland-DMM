import logging
import logging.config

logging.getLogger('tensorflow').disabled = True

###

console = None
file = None

filename = './logs/out.log'

###

# https://stackoverflow.com/a/56144390
logging.root.setLevel(logging.NOTSET)

# Create a custom logger
console = logging.getLogger('console')
file = logging.getLogger('file')

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(filename=filename)
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
