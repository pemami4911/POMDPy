__author__ = 'patrickemami'

import StepGenerator as Sg
import logging
import ActionSelectors
import RockAction


class TDStepper(Sg.StepGenerator):
    """
    This class implements the methods extend_and_backup, update_sequence, and get_step
    """
    def __init__(self, model):
        self.model = model
        self.logger = logging.getLogger('Model.TDStepper')
        self.ucb_coefficient = self.model.config["ucb_coefficient"]
        self.step_size = self.model.config["step_size"]
        self.algorithm = self.model.config["algorithm"]
        self.status = None

    def extend_and_backup(self, seq, maximum_depth):
        """
        The main functionality of this method is to extend the last entry of a history sequence to a terminal state
        by generating steps. This will populate the "immediate rewards" of the history sequence entries, which can
        then be used to back propagate the Q values
        :param history_sequence:
        :param maximum_depth:
        :return:
        """
        self.status = Sg.SearchStatus.UNINITIALIZED

        entries = seq.entry_sequence

        # start at the last entry, since we are extending the current sequence
        # for all intents and purposes, this is usually just the first and only entry in the sequence
        current_entry = entries[entries.__len__() - 1]
        current_node = current_entry.associated_belief_node

        if self.model.is_terminal(current_entry.state):
            self.logger.warning("Attempted to continue sequence from a terminal state")
            return Sg.SearchStatus.ERROR
        elif current_entry.action is not None:
            self.logger.warning("The last history entry in the sequence already has an action")
            return Sg.SearchStatus.ERROR
        elif current_entry.reward != 0:
            self.logger.warning("The last history entry in the sequence already has a nonzero reward")
            return Sg.SearchStatus.ERROR

        self.status = Sg.SearchStatus.INITIAL
        added = False

        while True:
            # Step the episode forward
            result, is_legal = self.get_step(current_entry, current_entry.state)

            # stop the search
            if result is None:
                self.status = Sg.SearchStatus.ERROR
                break

            ''' Advance the episode by creating a new Belief Node '''

            # Set the params of the current history entry using the ones we got from the result
            current_entry.reward = result.reward
            current_entry.action = result.action
            current_entry.observation = result.observation

            # If a goal state or you did an illegal action and blew up or ran out of steps you are done
            if result.is_terminal:
                self.status = Sg.SearchStatus.CLEAN_FINISH
                break
            elif not is_legal:
                self.status = Sg.SearchStatus.TERMINATED
                break
            elif current_node.depth >= maximum_depth:
                self.status = Sg.SearchStatus.OUT_OF_STEPS
                break

            # Create a new child belief node, and set the current node to be that node.
            next_node, added = current_node.create_or_get_child(current_entry.action, current_entry.observation)
            current_node = next_node

            current_entry = seq.add_entry()

            # Create a new history entry and step the history forward
            current_entry.register_entry(current_entry, current_node, result.next_state)

            if added:
                # If the current belief node was added (i.e. it didn't already exist within the belief tree),
                # Try to find a nearest neighbor of its parent belief node and see if it can be replaced by the
                # nearest neighbor in the belief tree. This encourages the reuse of belief nodes, which
                # encourages learning
                # Try to find the NN of the current node.
                nearest_belief_node = current_node.find_neighbor()

                if nearest_belief_node is not None:

                    print "NN found"

                    # Copy the old bin sequence, which contains the set of legal actions
                    old_bin_sequence = list(current_node.action_map.bin_sequence)

                    # Copy over the action map from the NN belief node
                    current_node.action_map = nearest_belief_node.action_map.copy()

                    # Reassign the old bin sequence
                    current_node.action_map.bin_sequence = old_bin_sequence

                    # Update the legality of the new action mapping entries with the old bin sequence
                    current_node.action_map.update_legality()

                    # Set the owner of the action map to be the current node
                    current_node.action_map.owner = current_node

        # If ya done run out of steps
        if self.status is Sg.SearchStatus.OUT_OF_STEPS:

            # Currently, the immediate reward is simply set to 0
            # Potentially find a way to get a better heuristic
            current_entry.reward = 0
            self.logger.warning("Ran out of steps!")
            self.status = Sg.SearchStatus.TERMINATED

        if self.status is Sg.SearchStatus.CLEAN_FINISH or self.status is Sg.SearchStatus.TERMINATED:

            self.update_sequence(entries)

        else:
            if self.status is Sg.SearchStatus.UNINITIALIZED:
                self.logger.warning("Smart stepper could not initialize")
            elif self.status is Sg.SearchStatus.INITIAL:
                self.logger.warning("Smart stepper exited while running")
            elif self.status is Sg.SearchStatus.ERROR:
                self.logger.warning("An error occurred while executing search algorithm")
            else:
                self.logger.warning("Invalid search status")

        return self.status

    def get_step(self, history_entry, state):
        """
        :param history_entry:
        :param state:
        :param historical_data:
        :return:
        """
        current_node = history_entry.associated_belief_node

        # The action to use when querying the black box model for the next step
        action = None

        # UCB1 action-selection
        if self.algorithm == "UCB":

            # Try all of the actions from the current belief
            action = ActionSelectors.expand_belief_node(current_node)

            # all actions have been attempted
            if action is None:

                action = ActionSelectors.ucb_action(current_node, self.ucb_coefficient)

                if action is None:
                    self.logger.warning("Node has no actions? Returning empty Step Result")
                    self.status = Sg.SearchStatus.ERROR
                    return None

        # Standard TD-Q-Learning
        elif self.algorithm == "TD":
            action = ActionSelectors.q_action(current_node)

        else:
            self.logger.fatal("No algorithm was specified in the config file")
            self.status = Sg.SearchStatus.ERROR

        # update the visit count for the action you just took
        current_node.action_map.update_entry_visit_count(action, 1)

        return self.model.generate_step(state, action)

    # Back-propagation of the new Q values
    def update_sequence(self, seq):
        if seq.__len__() < 1:
            self.logger.warning("Cannot update sequence of length < 1")
            return

        discount_factor = self.model.config["discount_factor"]
        step_size = self.model.config["step_size"]

        # reverse the sequence, since we must back up starting from the end
        r_seq = list(seq)
        r_seq.reverse()

        # Since the last history entry has no more actions to take, initialize maximal q to 0
        maximal_q = 0

        # handle the first entry in the reversed sequence, which won't have a corresponding observation mapping entry
        first_entry = r_seq[0]
        action_mapping_entry = first_entry.associated_belief_node.action_map.get_entry(first_entry.action)
        current_q = action_mapping_entry.mean_q_value

        # update the action mapping entry with the newly calculated maximal q value
        maximal_q = ((discount_factor * maximal_q) - current_q + first_entry.reward) * step_size
        action_mapping_entry.update_q_value(maximal_q)

        # traverse the rest of the list, starting with the second history entry
        for entry in r_seq[1:]:

            action_mapping_entry = entry.associated_belief_node.action_map.get_entry(entry.action)

            current_q = action_mapping_entry.mean_q_value

            maximal_q = ((discount_factor * maximal_q) - current_q + entry.reward) * step_size

            # update the action's Q value and visit count
            action_mapping_entry.update_q_value(maximal_q)

            # update the observation visit count
            obs_entry = action_mapping_entry.child_node.observation_map.get_entry(entry.observation)
            if obs_entry is None:
                print 'BREAK'
            obs_entry.update_visit_count(1)


