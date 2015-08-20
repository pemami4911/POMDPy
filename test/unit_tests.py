#!/usr/bin/env python
__author__ = 'patrickemami'

import random
import unittest
import os

import sys


par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
src_dir = os.path.join(par_dir, 'src')
sys.path.append(src_dir)

from POMDP.solvers.solver import Solver
from sampleproblems.rockproblem.rock_model import RockModel

model = RockModel("unit_tests")
solver = Solver(model)

''' --------- MCTS --------- '''
from POMDP.solvers.MCTS import MCTS
from actionselection import action_selectors

mcts = None

class MCTSTestCase(unittest.TestCase):

    def test_create_mcts(self):
        global mcts
        mcts = MCTS(solver, model)
        self.assertIsNotNone(mcts)

    def test_greedy_search(self):
        """
        Testing Greedy Search (Choosing action with the highest Q value)
        :return:
        """
        global mcts
        maximal = random.choice(mcts.policy.root.action_map.bin_sequence)
        mcts.policy.root.action_map.entries.get(maximal).update_q_value(1.0, 1)
        self.assertEqual(action_selectors.ucb_action(mcts, mcts.policy.root, True).bin_number, maximal)

    def test_ucb_search(self):
        """
        Testing UCB search
        :return:
        """
        global mcts

        # With equal Q values, action with the lowest count is selected by the UCB algorithm
        lowest_count_action = random.choice(mcts.policy.root.action_map.bin_sequence)
        for i in mcts.policy.root.action_map.bin_sequence:
            if i == lowest_count_action:
                mcts.policy.root.action_map.entries.get(i).update_visit_count(90)
            else:
                mcts.policy.root.action_map.entries.get(i).update_visit_count(100 + i)
            mcts.policy.root.action_map.entries.get(i).mean_q_value = 0.0
        self.assertEqual(action_selectors.ucb_action(mcts, mcts.policy.root, False).bin_number, lowest_count_action),

    def test_rollout_strategy(self):
        """
        Testing rollout strategy
        :return:
        """
        global mcts
        mcts.rollout_search()


if __name__ == '__main__':
    unittest.main()
