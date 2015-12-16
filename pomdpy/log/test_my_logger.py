__author__ = 'patrickemami'

import log_pomdpy
import logging

def test_logger():
    log_pomdpy.init_logger()
    logger = logging.getLogger('POMDPy.test')
    logger.debug("This is a test")