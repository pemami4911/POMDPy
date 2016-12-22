from __future__ import division
from .pickle_wrapper import load_pkl
from .plot_alpha_vectors import plot_alpha_vectors
import os
import time
from pomdpy.agent import Results
import numpy as np

my_dir = os.path.dirname(__file__)
weight_dir = os.path.join(my_dir, '..', '..', 'experiments', 'pickle_jar')


def plot_baseline(horizon, baseline):
    plot_alpha_vectors('Value Iteration - Planning Horizon - {}'.format(horizon), baseline, 3)


def eval_baseline(n_epochs, agent, horizon):
    n_repeats = 5
    random_action = False

    if horizon == 8:
        baseline = load_pkl(os.path.join(weight_dir, 'VI_planning_horizon_8.pkl'))
    elif horizon == 1:
        baseline = load_pkl(os.path.join(weight_dir, 'VI_planning_horizon_1.pkl'))
    elif horizon == 4:
        baseline = load_pkl(os.path.join(weight_dir, 'VI_planning_horizon_4.pkl'))
    elif horizon == 0:
        baseline = load_pkl(os.path.join(weight_dir, 'linear_alpha_net_vectors.pkl'))
    elif horizon == -1:
        baseline = None
        random_action = True
    else:
        raise ValueError('Unsupported baseline planning horizon')

    solver = agent.solver_factory(agent)
    model = agent.model

    experiment_results = Results()
    wrong_door_count = 0

    for repeats in range(n_repeats):
        np.random.seed(int(agent.model.seed) + 1)
        agent.model.seed = str(int(agent.model.seed) + 1)

        for epoch in range(n_epochs):
            epoch_start = time.time()
            model.reset_for_epoch()

            # evaluate agent
            reward = 0
            discounted_reward = 0
            discount = 1.0
            belief = model.get_initial_belief_state()

            while True:
                if random_action:
                    action = np.random.randint(model.num_actions)
                else:
                    action, v_b = solver.select_action(belief, baseline)

                step_result = model.generate_step(action)

                if not step_result.is_terminal:
                    belief = model.belief_update(belief, action, step_result.observation)

                if step_result.reward == -20.0:
                    wrong_door_count += 1

                reward += step_result.reward
                discounted_reward += discount * step_result.reward
                discount *= model.discount

                if step_result.is_terminal:
                    break

            experiment_results.time.add(time.time() - epoch_start)
            experiment_results.undiscounted_return.count += 1
            experiment_results.undiscounted_return.add(reward)
            experiment_results.discounted_return.count += 1
            experiment_results.discounted_return.add(discounted_reward)

    print('Results averaged over {} experiments with different random seeds\n'.format(n_repeats))
    print('epochs/experiment: {}'.format(model.n_epochs))
    print('undiscounted return/epoch: ' + str(experiment_results.undiscounted_return.mean) +
          ' std dev: ' + str(experiment_results.undiscounted_return.std_dev()))
    print('discounted return/epoch: ' + str(experiment_results.discounted_return.mean) +
          ' std dev: ' + str(experiment_results.discounted_return.std_dev()))
    print('time/epoch: ' + str(experiment_results.time.mean))
    print('wrong door count/experiment: {}'.format(wrong_door_count / n_repeats))
