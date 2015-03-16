__author__ = 'patrickemami'

import ActionMapping as Am
import ActionPool as Ap
import ActionNode as An
import numpy as np
import BeliefNode as Bn
import itertools

class DiscreteActionMapping(Am.ActionMapping):
    def __init__(self, belief_node_owner, discrete_action_pool, bin_sequence, preferred_actions):
        assert isinstance(belief_node_owner, Bn.BeliefNode)
        assert isinstance(discrete_action_pool, Ap.DiscreteActionPool)

        super(DiscreteActionMapping, self).__init__(belief_node_owner)
        self.pool = discrete_action_pool
        self.number_of_bins = self.pool.get_number_of_bins()
        self.entries = {}   # Dictionary of DiscreteActionMappingEntry objects
        self.bin_sequence = bin_sequence
        self.number_of_children = 0
        self.total_visit_count = 0

        for i in range(0,self.number_of_bins):
            entry = DiscreteActionMappingEntry()
            entry.bin_number = i
            entry.map = self
            entry.is_legal = False
            self.entries.__setitem__(i, entry)

        # Only entries in the sequence are legal
        for bin_number in self.bin_sequence:
            self.entries.get(bin_number).is_legal = True

        for i in preferred_actions:
            #self.entries.get(preferred_actions[i]).preferred_action = True
            self.entries.get(i).total_q_value = 0.0

    def copy(self):
        action_map_copy = DiscreteActionMapping(self.owner, self.pool, list(self.bin_sequence), [])
        action_map_copy.number_of_children = self.number_of_bins
        action_map_copy.entries = self.entries.copy()
        action_map_copy.number_of_children = self.number_of_children
        action_map_copy.total_visit_count = self.total_visit_count
        return action_map_copy

    def get_all_entries_q_values(self, action_map):
        assert action_map.entries.values().__len__() == self.entries.values().__len__()
        for entry1, entry2 in itertools.izip(self.entries.values(), action_map.entries.values()):
            entry2.mean_q_value = entry1.mean_q_value
            entry2.total_q_value = entry1.total_q_value
            entry2.visit_count = entry1.visit_count
        action_map.total_visit_count = self.total_visit_count
        return action_map

    def update_legality(self):
        for entry in self.entries.values():
            entry.is_legal = False

         # Only entries in the sequence are legal
        for bin_number in self.bin_sequence:
            self.entries.get(bin_number).is_legal = True

    def get_action_node(self, discrete_action):
        bin_number = discrete_action.get_bin_number()
        return self.entries.get(bin_number).child_node

    def create_action_node(self, discrete_action):
        bin_number = discrete_action.get_bin_number()
        entry = self.entries.get(bin_number)
        # the child of this entry
        # the parent of this action node is the entry
        entry.child_node = An.ActionNode(entry)
        self.number_of_children += 1
        return entry.child_node

    def delete_child(self, disc_entry):
        disc_entry.update(-disc_entry.visit_count, -disc_entry.total_q_value)
        disc_entry.child_node = None

    def get_child_entries(self):
        return_entries = []
        for i in range(0,self.number_of_bins):
            entry = self.entries.get(i)
            if entry.child_node is not None:
                return_entries.append(entry)
        return return_entries

    def get_visited_entries(self):
        return_entries = []
        for i in range(0, self.number_of_bins):
            entry = self.entries.get(i)
            if entry.visit_count > 0:
                return_entries.append(entry)
        return return_entries

    # Returns a list of all ActionMappingEntries associated with this mapping
    def get_all_entries(self):
        all_actions = self.entries.values()
        np.random.shuffle(all_actions)
        return all_actions

    def get_entry(self, d_action):
        return self.entries.get(d_action.get_bin_number())

    # No more bins to try -> no action to try
    # Otherwise we sample a new action using the first bin to be tried
    def get_next_action_to_try(self):
        unvisited_entries = []
        for entry in self.entries.values():
            # allow illegal entries
            #if entry.visit_count == 0:
            if entry.is_legal and entry.visit_count == 0:
                unvisited_entries.append(entry)
        if unvisited_entries.__len__() != 0:
            np.random.shuffle(unvisited_entries)
            return self.pool.sample_an_action(unvisited_entries[0].bin_number)
        else:
            return None

    def update_entry_visit_count(self, action, delta_n_visits):
        mapping_entry = self.get_entry(action)
        return mapping_entry.update_visit_count(delta_n_visits)

    def update(self):
        self.bin_sequence = self.pool.create_bin_sequence(self.owner)

        # reset the entries to false
        for entry in self.entries.values():
            entry.is_legal = False

         # Only entries in the sequence are legal
        for bin_number in self.bin_sequence:
            self.entries.get(bin_number).is_legal = True

class DiscreteActionMappingEntry(Am.ActionMappingEntry):
    """
    A concrete class implementing ActionMappingEntry for a discrete action space.

    Each entry stores its bin number and a reference back to its parent map, as well as a child node,
    visit count, total and mean Q-values, and a flag for whether or not the action is legal.
    """
    def __init__(self):
        self.bin_number = -1
        self.map = None     # DiscreteActionMapping
        self.child_node = None       # ActionNode
        self.visit_count = 0
        self.total_q_value = 0
        self.mean_q_value = 0
        self.is_legal = False

        # Mark this action mapping entry as preferred. This ensure that the Q value is always positive
        # So that the agent will favor this action, even if shit is hitting the fan
        self.preferred_action = False

    def get_action(self):
        return self.map.pool.sample_an_action(self.bin_number)

    # Update the action mapping entries visit count and the action maps total visit count
    def update_visit_count(self, delta_n_visits):
        if delta_n_visits == 0:
            return

        self.visit_count += delta_n_visits
        self.map.total_visit_count += delta_n_visits

        return self.visit_count

    def update_q_value(self, delta_total_q):
        if delta_total_q == 0:
            return False

        assert np.isfinite(delta_total_q)

        # Ensure that preferred actions never have negative Q values to favor them
        if self.preferred_action and delta_total_q < 0:
            delta_total_q = -delta_total_q

        # Add up the Q value
        self.total_q_value += delta_total_q

        # Update the mean Q
        old_mean_q = self.mean_q_value
        #if self.visit_count <= 0:
        #    self.mean_q_value = -np.inf
        #else:

        # Average the Q value by taking the Total Q value of this entry divided by the
        # number of times this action has been tried
        self.mean_q_value = self.total_q_value / self.map.total_visit_count

        return self.mean_q_value != old_mean_q

    def set_legal(self, legal):
        if not self.is_legal:
            if legal:
                self.is_legal = True
                if self.visit_count is 0:
                    self.map.bin_sequence.add(self.bin_number)
        else:
            if not self.is_legal:
                self.is_legal = False
                self.map.bin_sequence.remove(self.bin_number)






