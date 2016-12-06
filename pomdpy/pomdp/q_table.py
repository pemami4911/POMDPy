from __future__ import absolute_import
from builtins import range
from .belief_structure import BeliefStructure


class QTable(BeliefStructure):
    """
    Creates a Q table and visit frequency table for quick and dirty Q learning

    Indexing into the Q-table is done by

        q_table[action_idx][state_idx] = Q(s,a)

    <b>Only useful for fully-observable problems</b>

    """

    def __init__(self, agent):
        self.agent = agent
        self.q_table = None
        self.visit_frequency_table = None

    def reset(self):
        self.initialize()

    def initialize(self, init_value=None):
        """
        Create multidimensional tables of dim: num_observations x num_actions x num_states to
        store the estimated Q values and the visit frequency
        :param init_value - used to initialize the Q values to some arbitrary value
        :return:
        """
        self.q_table = [[init_value for _ in
                         range(self.agent.model.get_all_actions()[1])]
                        for _ in range(self.agent.model.get_all_states()[1])]

        self.visit_frequency_table = [[0 for _ in
                                       range(self.agent.model.get_all_actions()[1])]
                                      for _ in range(self.agent.model.get_all_states()[1])]