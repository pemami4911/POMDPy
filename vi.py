#!/usr/bin/env python
from __future__ import print_function
from pomdpy import Agent
from pomdpy.solvers import ValueIteration
from pomdpy.log import init_logger
from examples.tiger import TigerModel
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--env', type=str, help='Specify the env to solve {Tiger}')
    parser.add_argument('--solver', type=str,
                        help='Specify the solver to use {ValueIteration|LinearAlphaNet|VI-Baseline}')
    parser.add_argument('--seed', default=1993, type=int, help='Specify the random seed for numpy.random')
    parser.add_argument('--use_tf', dest='use_tf', action='store_true', help='Set if using TensorFlow')
    parser.add_argument('--discount', default=0.95, type=float, help='Specify the discount factor (default=0.95)')
    parser.add_argument('--n_epochs', default=1, type=int, help='Num of epochs of the experiment to conduct')
    parser.add_argument('--max_steps', default=10, type=int, help='Max num of steps per trial/episode/trajectory/epoch')
    parser.add_argument('--save', dest='save', action='store_true', help='Pickle the weights/alpha vectors')

    # Args for Deep Alpha Nets
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--learning_rate_minimum', default=0.0025, type=float)
    parser.add_argument('--learning_rate_decay', default=0.996, type=float)
    parser.add_argument('--learning_rate_decay_step', default=50, type=int)
    parser.add_argument('--beta', default=0.001, type=float, help='L2 regularization parameter')

    parser.add_argument('--test', default=10, type=int, help='Evaluate the agent every `test` epochs')
    parser.add_argument('--epsilon_start', default=0.02, type=float)
    parser.add_argument('--epsilon_minimum', default=0.05, type=float)
    parser.add_argument('--epsilon_decay', default=0.96, type=float)
    parser.add_argument('--epsilon_decay_step', default=75, type=int)
    parser.add_argument('--planning_horizon', default=5, type=int, help='Number of lookahead steps for value iteration')

    parser.set_defaults(use_tf=False)
    parser.set_defaults(save=False)

    args = vars(parser.parse_args())

    init_logger()

    np.random.seed(int(args['seed']))

    if args['solver'] == 'VI-Baseline':
        from experiments.scripts import approximate_vi_eval

        env = TigerModel(args)
        solver = ValueIteration
        agent = Agent(env, solver)
        approximate_vi_eval.eval_baseline(args['n_epochs'], agent, args['planning_horizon'])

    else:
        if args['solver'] == 'ValueIteration':
            solver = ValueIteration
        elif args['use_tf'] and args['solver'] == 'LinearAlphaNet':
            from pomdpy.solvers.linear_alpha_net import LinearAlphaNet
            solver = LinearAlphaNet
        else:
            raise ValueError('solver not supported')

        if args['env'] == 'Tiger':
            env = TigerModel(args)
            agent = Agent(env, solver)
            agent.discounted_return()
        else:
            print('Unknown env {}'.format(args['env']))
