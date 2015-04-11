__author__ = 'patrickemami'

import random
import logging
import json
import config_parser
import numpy as np

config = json.load(open(config_parser.sys_cfg, "r"))

''' ------------ GLOBAL VARS -------------'''
use_rave = False
rave_constant = 0.0

# controls the randomness of the action selection for TD- learning.
# Used for epsilon-greedy learning
epsilon = 1
k = 1
delta_k = config["delta_epsilon_k"]

''' ------------------ TD- Q-Learning ------------------ '''
def reset():
    global epsilon
    global k
    epsilon = 1
    k = 1

# TD-Q-Learning - grabs the action that has the highest expected q-value
# Default is epsilon-greedy
def q_action(current_node):
    """
    :param current_node:
    :return: action
    """
    logger = logging.getLogger("Model.q_action")
    max_q = -np.inf
    arm_to_play = None
    global epsilon
    global k
    global delta_k

    mapping = current_node.action_map

    # act randomly with decreasing probability 1/n, where n is the total visit count of the current action mapping
    if epsilon > np.random.uniform(0, 1):
        all_entries = mapping.get_all_entries()
        np.random.shuffle(all_entries)
        while True:
            if all_entries.__len__() == 0:
                logger.warning("No legal entries found when randomly trying an action. Death awaits")
                break
            if all_entries[0].is_legal:
                arm_to_play = all_entries[0].get_action()
                break
            else:
                all_entries = all_entries[1:]
    else:
        # Find the action from the set of all legal actions with the maximal Q value
        for entry in mapping.get_all_entries():

            # Ignore illegal actions
            if entry.is_legal:

                tmp_value = entry.mean_q_value

                if max_q < tmp_value:
                    max_q = tmp_value
                    arm_to_play = entry.get_action()

    # Decrease epsilon at a rate of 1/k
    k += delta_k
    epsilon = 1/k

    return arm_to_play

''' Multi-Armed Bandit '''

# UCB1 action selection algorithm
def ucb_action(mcts, current_node, greedy):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    N = mapping.total_visit_count
    log_n = np.log(N + 1)

    actions = mapping.get_all_entries()
    for action_entry in actions:

        # Skip illegal actions
        if not action_entry.is_legal:
            continue

        current_q = action_entry.mean_q_value

        # TODO RAVE stuff
        # if use_rave and action.visit_count > 0.0:

        # if has_alpha ...

        # TODO epsilon-greedy? Act randomly with probability 1/epsilon?
        # If the UCB coefficient is 0, this is just pure Q learning
        if not greedy:
            current_q += mcts.find_fast_ucb(N, action_entry.visit_count, log_n)

        if current_q >= best_q_value:
            if current_q > best_q_value:
                best_actions = []
            best_q_value = current_q
            # best actions is a list of Discrete Actions
            best_actions.append(action_entry.get_action())

    assert best_actions.__len__() is not 0

    return random.choice(best_actions)





