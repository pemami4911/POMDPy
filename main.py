#!/usr/bin/env python
from pomdpy import Agent
from pomdpy.solvers import POMCP, SARSA, ValueIteration
from pomdpy.log import init_logger
from examples.rock_problem import RockModel
from examples.tiger_problem import TigerModel
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--env', type=str, help='Specify the env to solve {RockProblem|TigerProblem}')
    parser.add_argument('--solver', type=str, help='Specify the solver to use {POMCP|SARSA|ValueIteration}')
    parser.add_argument('--seed', default=1993, type=int, help='Specify the random seed for numpy.random')
    parser.add_argument('--discount', default=0.95, type=float, help='Specify the discount factor (default=0.95)')
    parser.add_argument('--epsilon_start', default=0.2, type=float)
    parser.add_argument('--epsilon_end', default=0.1, type=float)
    parser.add_argument('--epsilon_decay', default=0.05, type=float)
    parser.add_argument('--max_steps', default=200, type=int, help='Max num of steps per episode')
    parser.add_argument('--n_sims', default=500, type=int, help='Num of MC sims to do at each belief node in MCTS')
    parser.add_argument('--n_runs', default=100, type=int, help='Num of runs of the experiment to conduct')
    parser.add_argument('--test_run', default=10, type=int, help='Evaluate the agent every `test_run` runs/episodes')
    parser.add_argument('--max_timeout', default=3600, type=int, help='Max num of sec the experiment should run before '
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
    args = parser.parse_args()

    init_logger()

    if args.solver == 'SARSA':
        solver = SARSA
    elif args.solver == 'POMCP':
        solver = POMCP
    elif args.solver == 'ValueIteration':
        solver = ValueIteration
    else:
        raise ValueError('solver not supported')

    if args.env == 'RockProblem':
        if isinstance(solver, ValueIteration):
            raise RuntimeError('Cannot run value iteration on RockSample problem')
        env = RockModel(args)
        env.draw_env()
        agent = Agent(env, solver)
        agent.discounted_return()
    elif args.env == 'TigerProblem':
        if isinstance(solver, POMCP):
            raise RuntimeError('Cannot run POMCP with Tiger problem')

        env = TigerModel(args)
        agent = Agent(env, solver)
        agent.discounted_return()
    else:
        print 'Unknown env %s' % args.env
