#!/usr/bin/env python
__author__ = 'patrickemami'

import random
import pytest
import os
import sys
from src.solvers import Solver, MCTS
from src.sample_problems.rock_problem import RockModel
from src.action_selection import ucb_action

par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
src_dir = os.path.join(par_dir, 'src')
sys.path.append(src_dir)
model = RockModel("unit_tests")
solver = Solver(model)

@pytest.fixture
def create_mcts():
    return MCTS(solver, model)


def test_create_mcts(create_mcts):
    mcts = create_mcts()
    assert mcts is not None


def test_greedy_search(create_mcts):
    """
    Testing Greedy Search (Choosing action with the highest Q value)
    :return:
    """
    mcts = create_mcts()
    maximal = random.choice(mcts.policy.root.action_map.bin_sequence)
    mcts.policy.root.action_map.entries.get(maximal).update_q_value(1.0, 1)
    assert ucb_action(mcts, mcts.policy.root, True).bin_number is maximal


def test_ucb_search(create_mcts):
    """
    Testing UCB search
    :return:
    """
    mcts = create_mcts()

    # With equal Q values, action with the lowest count is selected by the UCB algorithm
    lowest_count_action = random.choice(mcts.policy.root.action_map.bin_sequence)
    for i in mcts.policy.root.action_map.bin_sequence:
        if i == lowest_count_action:
            mcts.policy.root.action_map.entries.get(i).update_visit_count(90)
        else:
            mcts.policy.root.action_map.entries.get(i).update_visit_count(100 + i)
        mcts.policy.root.action_map.entries.get(i).mean_q_value = 0.0
    assert ucb_action(mcts, mcts.policy.root, False).bin_number is lowest_count_action


def test_rollout_strategy(create_mcts):
    """
    Testing rollout strategy
    :return:
    """
    mcts = create_mcts()
    mcts.rollout_search()


if __name__ == '__main__':
    pytest.main()
