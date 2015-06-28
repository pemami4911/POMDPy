__author__ = 'patrickemami'

import logging

import matplotlib.pyplot as plt
import numpy as np


'''
 For n runs, each containing 100 episodes, plot:
- the total accumulated reward for each episode
- Number of bad samples per episode
- Percent of actual good rocks that ended up being sampled per episode
- Number of illegal actions
- Number of steps each episode took
- Number of "good" checks vs. "bad" checks -- histogram of red/green bars
- frequency of each action

- Should have UCB vs. Standard TD-Q-Learning for each plot

- Do (30-100) Runs for UCB and TD. on
'''

import rock_model

my_model = rock_model.RockModel("rockproblem")
my_solver = RockSolver(my_model)
logger = logging.getLogger('POMDPy.results')

print "Trial for: ",
print my_model.config["algorithm"]

# Number of episodes per run
x = range(0, my_solver.n_episodes)

# Averages
average_reward_per_episode = [0 for _ in x]
average_good_checks_per_episode = [0 for _ in x]
average_bad_checks_per_episode = [0 for _ in x]
average_percent_good_rocks_sampled_per_episode = [0 for _ in x]
average_score = [0 for _ in x]
total_num_times_sampled = [0 for _ in x]

# Data points
num_bad_rocks_sampled = []
num_times_sampled = []
num_good_checks = []
num_bad_checks = []
percent_good_rocks_sampled = []
total_accumulated_rewards = []
score = []

RUNS = 100

for i in range(1, RUNS+1):

    num_bad_rocks_sampled.append([])
    num_times_sampled.append([])
    num_good_checks.append([])
    num_bad_checks.append([])
    percent_good_rocks_sampled.append([])
    total_accumulated_rewards.append([])
    score.append([])

    for policy, total_reward, num_reused_nodes in my_solver.generate_policy():
        for belief, action, reward in policy:
            # action.print_action()
            # belief.print_state()
            # print "Reward: ", reward
            # print "Num Reused Belief Nodes: ", num_reused_nodes
            continue
        # print "Unique rocks sampled: ", my_model.unique_rocks_sampled.values()
        # print "Actual rock states: ", my_model.actual_rock_states
        logger.info("Episode %s", my_solver.current_episode)
        num_bad_rocks_sampled[i-1].append(my_model.num_bad_rocks_sampled)
        num_good_checks[i-1].append(my_model.num_good_checks)
        num_bad_checks[i-1].append(my_model.num_bad_checks)
        total_accumulated_rewards[i-1].append(total_reward)
        percent_good_rocks_sampled[i-1].append(my_model.good_samples)
        num_times_sampled[i-1].append(my_model.num_times_sampled)
        score[i-1].append(float(my_model.unique_rocks_sampled.__len__())/float(sum(x > 0 for x in my_model.actual_rock_states)))
        logger.info("#GoodChecks: %s #BadChecks: %s #TotalReward: %s #NumGoodRocksSampled: %s",
                    str(my_model.num_good_checks), str(my_model.num_bad_checks), str(total_reward),
                    str(my_model.good_samples))
    '''
    print "Finished calculating policy: ", i+1
    print my_solver.policy.q_table
    print my_solver.policy.visit_frequency_table
    '''
    # Accumulate data points
    for i_ in x:
        average_reward_per_episode[i_] += total_accumulated_rewards[i-1][i_]
        average_bad_checks_per_episode[i_] += num_bad_checks[i-1][i_]
        average_good_checks_per_episode[i_] += num_good_checks[i-1][i_]
        average_percent_good_rocks_sampled_per_episode[i_] += percent_good_rocks_sampled[i-1][i_]
        total_num_times_sampled[i_] += num_times_sampled[i-1][i_]
'''
action_frequency = []
partial_sum = []
for i in range(my_solver.action_pool.get_number_of_bins()):
    for j in range(pow(2, my_model.n_rocks)):
        partial_sum.append(my_solver.policy.visit_frequency_table[j][i])
    action_frequency.append(np.sum(partial_sum))
    partial_sum = []
'''
# Calculate averages
# Divide through by the total number of times RAN/sampled to get the statistics
for i in x:
    average_percent_good_rocks_sampled_per_episode[i] /= float(total_num_times_sampled[i])
    average_reward_per_episode[i] /= float(RUNS)
    average_good_checks_per_episode[i] /= float(RUNS)
    average_bad_checks_per_episode[i] /= float(RUNS)

for j in range(RUNS):
    for i in x:
        average_score[i] = np.mean(score[j][i])

[plt.scatter(j, total_accumulated_rewards[i][j]) for j in x for i in range(RUNS)]
average_reward = plt.plot(average_reward_per_episode, color='orange', lw=3)
plt.xlabel('Episode #')
plt.title('Total Rewards per Episode over 100 Runs')
plt.show()

g_checks_plot = plt.bar(x, average_good_checks_per_episode, color='green')
b_checks_plot = plt.bar(x, average_bad_checks_per_episode, color='red')
plt.title('Frequency of Good/Bad use of Noisy Sensor over 100 Runs')
plt.xlabel('Episode #')
plt.show()

ave_score_plot = plt.bar(x, average_score, color='blue')
plt.title('Percent of Good Rocks sampled per Episode')
plt.xlabel('Episode #')
plt.ylabel('# of Unique, Good rocks sampled/Actual # of Good Rocks')
plt.show()
'''
action_freq_plot = plt.bar(np.arange(my_solver.action_pool.get_number_of_bins()), action_frequency)
plt.title('Frequency of each action')
plt.xlabel('Action bin-number')
plt.show()

sampling_plot = plt.bar(x, average_percent_good_rocks_sampled_per_episode)
plt.title('Average Percent of Rock Samples Receiving Good Rewards')
plt.xlabel('Episode #')
plt.ylim(0, 1.0)
plt.show()
'''