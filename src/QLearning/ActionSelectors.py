__author__ = 'patrickemami'

import numpy as np
import logging
import parser, json

config = json.load(open(parser.cfg_file, "r"))


# controls the randomness of the action selection for TD- learning.
# Used for epsilon-greedy learning
epsilon = config["epsilon"]

def expand_belief_node(current_node):
    """
    Expand all of the actions from a belief node using regular Q-Learning. Once all of the actions have been tried,
    use UCB1
    :param current_entry:
    :param state:
    :param model:
    :return:
    """
    mapping = current_node.action_map

    # Actions that have been tried are removed from the mapping's bin sequence
    # get_next_action_to_try returns a random untried action
    if mapping.get_next_action_to_try() is None:
        return None
    else:
        return q_action(current_node)

# UCB1 action selection algorithm
def ucb_action(current_node, ucb_coefficient):
    """
    :param current_node:
    :param ucb_coefficient:
    :return: action
    """
    logger = logging.getLogger("Model.ucb_action")
    max_ucb_value = -np.inf
    mapping = current_node.action_map
    arm_to_play = None

    for entry in mapping.get_visited_entries():

        # Ignore illegal actions
        if entry.is_legal:
            tmp_value = entry.mean_q_value + ucb_coefficient \
                                    * np.sqrt(2.0 * np.log(mapping.total_visit_count)/entry.visit_count)

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
def q_action(current_node):
    """
    :param current_node:
    :return: action
    """
    logger = logging.getLogger("Model.q_action")
    max_q = -np.inf
    arm_to_play = None
    global epsilon

    mapping = current_node.action_map

    # act randomly with decreasing probability 1/n, where n is the total visit count of the current action mapping
    if mapping.total_visit_count > 0 and epsilon > np.random.uniform(0, 1):
        all_entries = mapping.get_all_entries()
        np.random.shuffle(all_entries)
        while True:
            if all_entries.__len__() == 0:
                logger.warning("No legal entries found when randomly trying an action. Death awaits")
                break
            random_action_mapping_entry = all_entries[0]
            if random_action_mapping_entry.is_legal:
                print "Acted randomly"
                return random_action_mapping_entry.get_action()
            else:
                all_entries = all_entries[1:]


    # Find the action from the set of all legal actions with the maximal Q value
    for entry in mapping.get_all_entries():

        # Ignore illegal actions
        if entry.is_legal:

            tmp_value = entry.mean_q_value

            if max_q < tmp_value:
                max_q = tmp_value
                arm_to_play = entry.get_action()

    epsilon *= 0.9999
    return arm_to_play
