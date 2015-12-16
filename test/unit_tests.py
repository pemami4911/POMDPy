#!/usr/bin/env python
__author__ = 'patrickemami'

from pomdpy import Agent
from pomdpy.solvers import MCTS
from examples.rock_problem import RockModel
from pomdpy.action_selection import ucb_action
import random
import unittest

model = RockModel("unit_tests")
agent = Agent(model, MCTS)


class TestPOMDPy(unittest.TestCase):

    def setUp(self, *args):
        self.solver = agent.solver_factory(agent, model)

    def test_instantiate_solver(self):
        assert self.solver is not None

    def test_greedy_search(self):
        """
        Testing Greedy Search (Choosing action with the highest Q value)
        :return:
        """
        maximal = random.choice(self.solver.policy.root.action_map.bin_sequence)
        self.solver.policy.root.action_map.entries.get(maximal).update_q_value(1.0, 1)
        assert ucb_action(self.solver, self.solver.policy.root, True).bin_number is maximal

    def test_ucb_search(self):
        """
        Testing UCB search
        :return:
        """
        # With equal Q values, action with the lowest count is selected by the UCB algorithm
        lowest_count_action = random.choice(self.solver.policy.root.action_map.bin_sequence)
        for i in self.solver.policy.root.action_map.bin_sequence:
            if i == lowest_count_action:
                self.solver.policy.root.action_map.entries.get(i).update_visit_count(90)
            else:
                self.solver.policy.root.action_map.entries.get(i).update_visit_count(100 + i)
            self.solver.policy.root.action_map.entries.get(i).mean_q_value = 0.0
        assert ucb_action(self.solver, self.solver.policy.root, False).bin_number is lowest_count_action

    def test_rollout_strategy(self):
        """
        Testing rollout strategy
        :return:
        """
        model.reset_for_run()
        self.solver.rollout_search()


if __name__ == '__main__':
    unittest.main()
