#!/usr/bin/env python
__author__ = 'patrickemami'

import sys
from pomdpy import Agent
from pomdpy.solvers import MCTS
from pomdpy.log import init_logger
from examples.rock_problem import RockModel
from examples.tiger_problem import TigerModel

SAMPLE_PROBLEM = sys.argv[1]

if __name__ == '__main__':

    init_logger()

    if SAMPLE_PROBLEM == "1":
        simulator = RockModel("Rock Problem")
        simulator.draw_env()
        agent = Agent(simulator, MCTS)
        agent.discounted_return()
        agent.logger.info("Map: " + simulator.rock_config["map_file"])
    elif SAMPLE_PROBLEM == "2":
        simulator = TigerModel("Tiger Problem")
        agent = Agent(simulator, MCTS)
        agent.discounted_return()
    else:
        print "Unable to execute unknown sample problem " + SAMPLE_PROBLEM


