#!/usr/bin/env python
from __future__ import print_function
from pomdpy import Agent
from pomdpy.solvers import POMCP, SARSA, ValueIteration, LinearAlphaNet
from pomdpy.log import init_logger
from examples.rock_sample import RockModel
from examples.tiger import TigerModel
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--env', type=str, help='Specify the env to solve {RockSample|Tiger}')
    parser.add_argument('--solver', type=str,
                        help='Specify the solver to use {POMCP|SARSA|ValueIteration|LinearAlphaNet}')
    parser.add_argument('--seed', default=1993, type=int, help='Specify the random seed for numpy.random')
    parser.add_argument('--use_tf', dest='use_tf', action='store_true', help='Set if using TensorFlow')
    parser.add_argument('--discount', default=0.95, type=float, help='Specify the discount factor (default=0.95)')
    parser.add_argument('--n_epochs', default=100, type=int, help='Num of epochs of the experiment to conduct')
    parser.add_argument('--max_steps', default=200, type=int, help='Max num of steps per trial/episode/trajectory/epoch')
    # Args for Deep Alpha Nets
    # TODO: Hyper-parameter search needed
    parser.add_argument('--learning_rate', default=0.0025, type=float)
    parser.add_argument('--learning_rate_minimum', default=0.0025, type=float)
    parser.add_argument('--learning_rate_decay', default=0.96, type=float)
    parser.add_argument('--learning_rate_decay_step', default=10, type=int)

    # Args for value iteration
    parser.add_argument('--planning_horizon', default=5, type=int, help='Number of lookahead steps for value iteration')

    # Args for POMCP/SARSA
    parser.add_argument('--test', default=10, type=int, help='Evaluate the agent every `test` epochs')
    parser.add_argument('--epsilon_start', default=0.2, type=float)
    parser.add_argument('--epsilon_end', default=0.1, type=float)
    parser.add_argument('--epsilon_decay', default=0.05, type=float)
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
    args = vars(parser.parse_args())

    init_logger()

    if args['solver'] == 'SARSA':
        solver = SARSA
    elif args['solver'] == 'POMCP':
        solver = POMCP
    elif args['solver'] == 'ValueIteration':
        solver = ValueIteration
    elif args['solver'] == 'LinearAlphaNet':
        solver = LinearAlphaNet
    else:
        raise ValueError('solver not supported')

    if args['env'] == 'RockSample':
        if not (issubclass(solver, SARSA) or issubclass(solver, POMCP)):
            raise RuntimeError('Solver not supported on RockSample')
        env = RockModel(args)
        env.draw_env()
        agent = Agent(env, solver)
        agent.discounted_return()

    elif args['env'] == 'Tiger':
        if not (issubclass(solver, ValueIteration) or issubclass(solver, LinearAlphaNet)):
            raise RuntimeError('Solver not supported on Tiger example')

        env = TigerModel(args)
        agent = Agent(env, solver)
        agent.discounted_return()
    else:
        print('Unknown env {}'.format(args['env']))
