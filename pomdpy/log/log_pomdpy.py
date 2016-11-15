import logging
import logging.handlers
from pomdpy.util import config_parser


def init_logger():
    my_logger = logging.getLogger('POMDPy')

    # default log format has time, module name, and message
    my_format = "%(asctime)s - %(name)s - %(message)s"

    # 5 files, 10 MB each
    my_handler = logging.handlers.RotatingFileHandler(
        filename=config_parser.log_path,
        maxBytes=10000000,
        backupCount=4)

    # set the format
    my_handler.setFormatter(logging.Formatter(my_format))

    # set default log level
    my_logger.setLevel(logging.DEBUG)

    # add a handler to the logger
    my_logger.addHandler(my_handler)
