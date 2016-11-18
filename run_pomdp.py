#!/usr/bin/env python

from pomdpy import Agent
from pomdpy.solvers import POMCP
from pomdpy.solvers import SARSA
from pomdpy.log import init_logger
from examples.rock_problem import RockModel
from examples.tiger_problem import TigerModel
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--env', type=str, help='Specify the env to solve {RockProblem|TigerProblem}')
    parser.add_argument('--solver', type=str, help='Specify the solver to use {POMCP|SARSA}')
    args = parser.parse_args()

    init_logger()

    if args.solver == 'SARSA':
        solver = SARSA
    elif args.solver == 'POMCP':
        solver = POMCP
    else:
        raise ValueError('solver not supported')

    if args.env == 'RockProblem':
        env = RockModel(args.env)
        env.draw_env()
        agent = Agent(env, solver)
        agent.discounted_return()
        agent.logger.info('Map: ' + env.rock_config["map_file"])

    elif args.env == 'TigerProblem':
        env = TigerModel(args.env)
        agent = Agent(env, solver)
        agent.discounted_return()
    else:
        print 'Unknown env %s' % args.problem
