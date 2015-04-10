__author__ = 'patrickemami'

import unittest
import RockModel
import Solver


model = RockModel.RockModel("UnitTest")
solver = Solver.Solver(model)

''' --------- MCTS --------- '''
import MCTS
import random
import ActionSelectors

mcts = None

class MCTSTestCase(unittest.TestCase):

    def test_create_mcts(self):
        global mcts
        mcts = MCTS.MCTS(solver, model)
        self.assertIsNotNone(mcts)

    def test_greedy_search(self):
        """
        Testing Greedy Search (Choosing action with the highest Q value)
        :return:
        """
        global mcts
        maximal = random.choice(mcts.policy.root.action_map.bin_sequence)
        mcts.policy.root.action_map.entries.get(maximal).update_q_value(1.0, 1)
        self.assertEqual(ActionSelectors.ucb_action(mcts, mcts.policy.root, True).get_bin_number(), maximal)

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
                mcts.policy.root.action_map.entries.get(i).update_visit_count(99)
            else:
                mcts.policy.root.action_map.entries.get(i).update_visit_count(100 + i)
            mcts.policy.root.action_map.entries.get(i).mean_q_value = 0.0
        self.assertEqual(ActionSelectors.ucb_action(mcts, mcts.policy.root, False).get_bin_number(), lowest_count_action)


if __name__ == '__main__':
    unittest.main()
