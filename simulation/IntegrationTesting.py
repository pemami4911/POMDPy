__author__ = 'patrickemami'


import RockSolver
import RockModel

num_trials = 100
average_reward = 0
average_step_count = 0
average_successful_samples = 0

my_model = RockModel.RockModel("RockProblem")
my_solver = RockSolver.RockSolver(my_model)

for i in range(0, num_trials):
    total_reward, step_count, samples = my_solver.generate_policy()
    average_reward = (average_reward + total_reward) / (i+1)
    average_step_count = (average_step_count + step_count) / (i+1)
    average_successful_samples = (average_successful_samples + samples) / (i+1)

print "System configurations: "
print "Map: ", my_model.config["map_file"]
print "Algorithm: ", my_model.config["algorithm"]
print "Number of episodes per policy generation: ", my_model.config["num_episodes"]
print "Maximum depth allowed for belief tree: ", my_model.config["maximum_depth"]
print "After ", num_trials, " trials, the average reward was: ",
print average_reward
print "The average step count was: ",
print average_step_count
print "The average # of times the agent successfully sampled a rock was: ",
print average_successful_samples