__author__ = 'patrickemami'


import RockSolver
import RockModel

my_model = RockModel.RockModel("RockProblem")

my_solver = RockSolver.RockSolver(my_model)

policy, step_count, samples = my_solver.generate_policy()

for i in policy:
    i.print_action()

print "Successful samples: ", samples
print "Unique rocks sampled: ", my_model.unique_rocks_sampled.values()
