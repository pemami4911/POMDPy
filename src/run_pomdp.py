#!/usr/bin/env python
__author__ = 'patrickemami'

"""
Run POMDPy for Rock Problem
"""
from POMDP.Solvers import *
from RockProblem import *

if __name__ == '__main__':
    simulator = RockModel.RockModel("Rock Problem")
    simulator.draw_env()
    my_solver = Solver.Solver(simulator)
    my_solver.discounted_return()

