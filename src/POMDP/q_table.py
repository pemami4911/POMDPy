__author__ = 'patrickemami'

import belief_structure

class QTable(belief_structure.BeliefStructure):
    """
    Creates a Q table and visit frequency table for quick and dirty Q learning
    """
    def __init__(self, solver):
        self.solver = solver
        self.q_table = None
        self.visit_frequency_table = None

    def reset(self):
        self.initialize()

    def initialize(self):
        self.q_table = [[None for _ in range(self.solver.action_pool.all_actions.__len__())]
                        for _ in range(self.solver.model.num_states)]
        self.visit_frequency_table = [[0 for _ in range(self.solver.action_pool.all_actions.__len__())]
                                        for _ in range(self.solver.model.num_states)]