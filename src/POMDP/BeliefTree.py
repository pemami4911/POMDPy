__author__ = 'patrickemami'

import BeliefNode
import logging
import BeliefStructure

class BeliefTree(BeliefStructure.BeliefStructure):
    """
    Contains the BeliefTree class, which represents an entire belief tree.
    *
    * Most of the work is done in the individual classes for the mappings and nodes; this class
    * simply owns a root node, and keeps track of a vector of all of the nodes in the entire tree
    * for convenient iteration and serialization.
    """

    def __init__(self, solver):
        super(BeliefTree, self).__init__()
        self.logger = logging.getLogger('Model.BeliefTree')
        self.solver = solver
        self.root = None

    def prune_tree(self):
        """
        Clears out the tree. This is difficult, because every node has a reference to its owner and child.
        :return:
        """



    # --------- TREE MODIFICATION ------- #
    def reset(self):
        self.prune_tree()
        self.root = BeliefNode.BeliefNode(self.solver, None, None)
        return self.root

    def reset_root_data(self):
        self.root.data = self.solver.model.create_root_historical_data(self.solver)

    def initialize(self):
        self.reset_root_data()
        self.root.action_map = self.solver.action_pool.create_action_mapping(self.root)


