#!/usr/bin/env python
__author__ = 'patrickemami'

"""
Run POMDPy for Rock Problem
"""
from POMDP.Solvers import *
from RockProblem import *
from TigerProblem import *

SAMPLE_PROBLEM = 1

if __name__ == '__main__':

    if SAMPLE_PROBLEM == 1:
        simulator = RockModel.RockModel("Rock Problem")
        simulator.draw_env()
        my_solver = Solver.Solver(simulator)
        my_solver.discounted_return()
    elif SAMPLE_PROBLEM == 2:
        simulator = TigerModel.TigerModel("Tiger Problem")
        my_solver = Solver.Solver(simulator)
        my_solver.discounted_return()
