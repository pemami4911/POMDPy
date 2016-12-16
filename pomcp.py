#!/usr/bin/env python
from __future__ import print_function
from pomdpy import Agent
from pomdpy.solvers import POMCP
from pomdpy.log import init_logger
from examples.rock_sample import RockModel
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--env', type=str, help='Specify the env to solve {RockSample}')
    parser.add_argument('--solver', default='POMCP', type=str,
                        help='Specify the solver to use {POMCP}')
    parser.add_argument('--seed', default=1993, type=int, help='Specify the random seed for numpy.random')
    parser.add_argument('--use_tf', dest='use_tf', action='store_true', help='Set if using TensorFlow')
    parser.add_argument('--discount', default=0.95, type=float, help='Specify the discount factor (default=0.95)')
    parser.add_argument('--n_epochs', default=100, type=int, help='Num of epochs of the experiment to conduct')
    parser.add_argument('--max_steps', default=200, type=int, help='Max num of steps per trial/episode/trajectory/epoch')
    parser.add_argument('--save', dest='save', action='store_true', help='Pickle the weights/alpha vectors')
    parser.add_argument('--test', default=10, type=int, help='Evaluate the agent every `test` epochs')
    parser.add_argument('--epsilon_start', default=0.5, type=float)
    parser.add_argument('--epsilon_minimum', default=0.1, type=float)
    parser.add_argument('--epsilon_decay', default=0.95, type=float)
    parser.add_argument('--epsilon_decay_step', default=20, type=int)
    parser.add_argument('--n_sims', default=500, type=int,
                        help='For POMCP, this is the num of MC sims to do at each belief node. '
                             'For SARSA, this is the number of rollouts to do per epoch')
    parser.add_argument('--timeout', default=3600, type=int, help='Max num of sec the experiment should run before '
                                                                  'timeout')
    parser.add_argument('--preferred_actions', dest='preferred_actions', action='store_true', help='For RockSample, '
                                                    'specify whether smart actions should be used')
    parser.add_argument('--ucb_coefficient', default=3.0, type=float, help='Coefficient for UCB algorithm used by MCTS')
    parser.add_argument('--n_start_states', default=2000, type=int, help='Num of state particles to generate for root '
                        'belief node in MCTS')
    parser.add_argument('--min_particle_count', default=1000, type=int, help='Lower bound on num of particles a belief '
                        'node can have in MCTS')
    parser.add_argument('--max_particle_count', default=2000, type=int, help='Upper bound on num of particles a belief '
                        'node can have in MCTS')
    parser.add_argument('--max_depth', default=100, type=int, help='Max depth for a DFS of the belief search tree in '
                        'MCTS')
    parser.add_argument('--action_selection_timeout', default=60, type=int, help='Max num of secs for action selection')

    parser.set_defaults(preferred_actions=False)
    parser.set_defaults(use_tf=False)
    parser.set_defaults(save=False)

    args = vars(parser.parse_args())

    init_logger()

    np.random.seed(args['seed'])

    if not args['solver'] == 'POMCP':
        raise ValueError('solver not supported')
    else:
        solver = POMCP

    if args['env'] == 'RockSample':
        env = RockModel(args)
        env.draw_env()
        agent = Agent(env, solver)
        agent.discounted_return()
    else:
        print('Unknown env {}'.format(args['env']))
