__author__ = 'patrickemami'

import numpy as np
import logging
import config_parser
import json

config = json.load(open(config_parser.cfg_file, "r"))

# controls the randomness of the action selection for TD- learning.
# Used for epsilon-greedy learning
epsilon = 1
k = 1
delta_k = config["delta_epsilon_k"]

def reset():
    global epsilon
    global k
    epsilon = 1
    k = 1

def expand_belief_node(belief_node, history_entry):
    """
    Expand all of the actions from a belief node using regular Q-Learning. Once all of the actions have been tried,
    use UCB1
    :param current_entry:
    :param state:
    :param model:
    :return:
    """

    # Actions that have been tried are removed from the mapping's bin sequence
    # get_next_action_to_try returns a random untried action
    if np.count_nonzero(belief_node.solver.policy.visit_frequency_table[history_entry.state.hash()][:]) !=\
            belief_node.action_map.number_of_bins:
        return q_action(belief_node)
    else:
        return None

# UCB1 action selection algorithm
def ucb_action(current_node, history_entry, ucb_coefficient):
    """
    :param current_node:
    :param ucb_coefficient:
    :return: action
    """
    logger = logging.getLogger("Model.ucb_action")
    max_ucb_value = -np.inf
    mapping = current_node.action_map
    arm_to_play = None

    for entry in mapping.get_all_entries():

        # Ignore illegal actions
        if entry.is_legal:
            tmp_value = entry.mean_q_value + ucb_coefficient \
                * np.sqrt(2.0 * np.log(np.sum(current_node.solver.policy.visit_frequency_table[history_entry.state.hash()][:]))
                / current_node.solver.policy.visit_frequency_table[history_entry.state.hash()][entry.bin_number])

            if not np.isfinite(tmp_value):
                logger.warning("Infinite/NaN value when calculating ucb action")

            if max_ucb_value < tmp_value:
                max_ucb_value = tmp_value
                # get the action corresponding to this mapping entry
                arm_to_play = entry.get_action()

    if arm_to_play is None:
        logger.warning("Couldn't find any action to take.. in ucb_action")

    return arm_to_play

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
                print "Acted randomly"
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
