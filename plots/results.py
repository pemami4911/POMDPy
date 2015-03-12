__author__ = 'patrickemami'

import numpy as np
import matplotlib.pyplot as plt

'''
 For n runs, each containing 100 episodes, plot:
- the total accumulated reward for each episode
- the percent success for each episode, i.e. the weighted sum of the percent successful samples and the points
-       acquired from sampling vs. the best possible amount of points to be had from sampling for that episode
- Number of steps each episode took
- Immediate reward for each step of each episode
- frequency of each action
- # of bad rocks sampled

- Should have UCB vs. Standard TD-Q-Learning for each plot
'''

import RockSolver
import RockModel

my_model = RockModel.RockModel("RockProblem")
my_solver = RockSolver.RockSolver(my_model)

RUNS = 5

for i in range(RUNS):
    for policy, total_reward, num_reused_nodes in my_solver.generate_policy():
        for belief, action, reward in policy:
            #action.print_action()
            #belief.print_state()
            #print "Reward: ", reward
            #print "Num Reused Belief Nodes: ", num_reused_nodes
            continue
        #print "Unique rocks sampled: ", my_model.unique_rocks_sampled.values()
        #print "Actual rock states: ", my_model.actual_rock_states
    print "Finished calculating policy: ", i

total_rewards_series = my_solver.total_accumulated_rewards

rewards_plot = plt.plot(total_rewards_series)
plt.xlabel('Episode #')
plt.title('Total Accumulated Reward per Episode')
plt.show()
