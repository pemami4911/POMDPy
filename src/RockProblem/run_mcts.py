__author__ = 'patrickemami'

import RockModel
import Solver
import objgraph

simulator = RockModel.RockModel("Rock Problem")

my_solver = Solver.Solver(simulator)

my_solver.discounted_return()

