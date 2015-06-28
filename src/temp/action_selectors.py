__author__ = 'patrickemami'

import random
import json
import numpy as np
import config_parser

config = json.load(open(config_parser.sys_cfg, "r"))

''' ------------ GLOBAL VARS -------------'''
use_rave = False
rave_constant = 0.0

# UCB1 action selection algorithm
def ucb_action(mcts, current_node, greedy):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    N = mapping.total_visit_count
    log_n = np.log(N + 1)

    actions = mapping.entries.values()
    random.shuffle(actions)
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

# Add more action selectors here




