#!/usr/bin/env python

__author__ = 'patrickemami'

import random
import unittest
import os
import sys
from pomdpy.solvers.MCTS import MCTS
from pomdpy.solvers import Solver
from pomdpy.examples.rock_problem import RockModel
from pomdpy.action_selection import ucb_action

par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
src_dir = os.path.join(par_dir, 'src')
sys.path.append(src_dir)

model = RockModel("unit_tests")
solver = Solver(model)


class TestPOMDPY(unittest.TestCase):

    def setUp(self, *args):
        self.mcts = MCTS(solver, model)

    def test_create_mcts(self):
        assert self.mcts is not None


    def test_greedy_search(self):
        """
        Testing Greedy Search (Choosing action with the highest Q value)
        :return:
        """
        maximal = random.choice(self.mcts.policy.root.action_map.bin_sequence)
        self.mcts.policy.root.action_map.entries.get(maximal).update_q_value(1.0, 1)
        assert ucb_action(self.mcts, self.mcts.policy.root, True).bin_number is maximal


    def test_ucb_search(self):
        """
        Testing UCB search
        :return:
        """
        # With equal Q values, action with the lowest count is selected by the UCB algorithm
        lowest_count_action = random.choice(self.mcts.policy.root.action_map.bin_sequence)
        for i in self.mcts.policy.root.action_map.bin_sequence:
            if i == lowest_count_action:
                self.mcts.policy.root.action_map.entries.get(i).update_visit_count(90)
            else:
                self.mcts.policy.root.action_map.entries.get(i).update_visit_count(100 + i)
            self.mcts.policy.root.action_map.entries.get(i).mean_q_value = 0.0
        assert ucb_action(self.mcts, self.mcts.policy.root, False).bin_number is lowest_count_action


    def test_rollout_strategy(self):
        """
        Testing rollout strategy
        :return:
        """
        model.reset_for_run()
        self.mcts.rollout_search()


if __name__ == '__main__':
    unittest.main()
