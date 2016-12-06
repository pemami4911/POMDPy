from __future__ import print_function
from builtins import object
from pomdpy.util import print_divider


class HistoryEntry(object):
    """
    * Contains the HistoryEntry class, which represents a single entry in a history sequence.
    *
    * The core functionality of the entry is to store a state, action, observation and reward
    * tuple (s, a, o, r, s')
    """

    def __init__(self, owning_sequence, id):
        self.owning_sequence = owning_sequence
        self.associated_belief_node = None
        self.id = id
        self.state = None
        self.action = None
        self.observation = None
        self.reward = 0

    def register_node(self, node):
        if self.associated_belief_node is node:
            return

        if self.associated_belief_node is not None:
            self.associated_belief_node.remove_particle(self)
            self.associated_belief_node = None
        if node is not None:
            self.associated_belief_node = node
            self.associated_belief_node.add_particle(self)

    def register_state(self, state):
        if self.state is state:
            return

        if self.state is not None:
            self.state = None
        if state is not None:
            self.state = state

    @staticmethod
    def register_entry(current_entry, node, state):
        current_entry.register_state(state)
        current_entry.register_node(node)

    @staticmethod
    def update_history_entry(h, r, a, o, s):
        h.reward = r
        h.action = a
        h.observation = o
        h.register_entry(h, None, s)


class HistorySequence(object):
    """
    Represents a single history sequence.
    *
    * The sequence owns its entries, which are stored in entry_sequence
    """

    def __init__(self, id):
        self.id = id
        self.entry_sequence = []

    def get_states(self):
        states = []
        for i in self.entry_sequence:
            states.append(i.state)
        return states

    def get_length(self):
        return self.entry_sequence.__len__()

    # adds a new entry to this sequence, and returns a pointer to it.
    def add_entry(self):
        new_entry = HistoryEntry(self, self.entry_sequence.__len__())
        self.entry_sequence.append(new_entry)
        return new_entry

    def remove_entry(self, history_entry):
        del self.entry_sequence[history_entry.id]

    def show(self):
        print_divider("medium")
        print("\tDisplaying history sequence")
        for entry in self.entry_sequence:
            print_divider("medium")
            print("id: ", entry.id)
            print("action: ", entry.action.to_string())
            print("observation: ", entry.observation.to_string())
            print("next state: ", entry.state.to_string())
            print("reward: ", entry.reward)


class Histories(object):
    """
    Owns a collection of history sequences.
    *
    * The createSequence() method is the usual way to make a new history sequence, as it will
    * be owned by this Histories object.
    """

    def __init__(self):
        self.sequences_by_id = []

    def get_number_of_sequences(self):
        return self.sequences_by_id.__len__()

    def create_sequence(self):
        hist_seq = HistorySequence(self.sequences_by_id.__len__())
        self.sequences_by_id.append(hist_seq)
        return hist_seq

    def delete_sequence(self, sequence):
        seq_id = sequence.id

        self.sequences_by_id[seq_id] = self.sequences_by_id.pop()
        self.sequences_by_id[seq_id].id = seq_id
