__author__ = 'patrickemami'


import RockSolver
import RockModel

my_model = RockModel.RockModel("RockProblem")

my_solver = RockSolver.RockSolver(my_model)

policy, step_count = my_solver.generate_policy()

for i, j in policy:
    i.print_action()
    j.print_state()

print "Unique rocks sampled: ", my_model.unique_rocks_sampled.values()
print "Actual rock states: ", my_model.actual_rock_states

