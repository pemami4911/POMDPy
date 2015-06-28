#!/usr/bin/env python
__author__ = 'patrickemami'

"""
Run POMDPy for Rock Problem
"""
from POMDP.solvers.solver import Solver
from sampleproblems.rockproblem.rock_model import RockModel
from sampleproblems.tigerproblem.tiger_model import TigerModel
from log.log_pomdpy import init_logger

SAMPLE_PROBLEM = 1

if __name__ == '__main__':

    init_logger()

    if SAMPLE_PROBLEM == 1:
        simulator = RockModel("Rock Problem")
        simulator.draw_env()
        my_solver = Solver(simulator)
        my_solver.discounted_return()
    elif SAMPLE_PROBLEM == 2:
        simulator = TigerModel("Tiger Problem")
        my_solver = Solver(simulator)
        my_solver.discounted_return()
