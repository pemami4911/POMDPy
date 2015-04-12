__author__ = 'patrickemami'

import BeliefNode
import BeliefStructure

class BeliefTree(BeliefStructure.BeliefStructure):
    """
    Contains the BeliefTree class, which represents an entire belief tree.
    *
    * Most of the work is done in the individual classes for the mappings and nodes; this class
    * simply owns a root node and handles pruning
    """

    def __init__(self, solver):
        super(BeliefTree, self).__init__()
        self.solver = solver
        self.root = None

    def prune_tree(self, belief_tree):
        """
        Clears out a belief tree. This is difficult, because every node has a reference to its owner and child.
        :return:
        """
        self.prune_node(belief_tree.root)
        belief_tree.root = None

    def prune_node(self, belief_node):
        if belief_node is None:
            return

        belief_node.parent_entry = None
        belief_node.action_map.owner = None
        # self.solver.action_pool.remove_mapping(belief_node.data, belief_node.action_map)
        action_mapping_entries = belief_node.action_map.get_child_entries()
        for entry in action_mapping_entries:
            # Action Node
            entry.child_node.parent_entry = None
            entry.map = None
            entry.child_node.observation_map.owner = None
            for observation_entry in entry.child_node.observation_map.child_map.values():
                self.prune_node(observation_entry.child_node)
                observation_entry.map = None
                observation_entry.child_node = None
            entry.child_node.observation_map = None
            entry.child_node = None
        belief_node.action_map = None

    # --------- TREE MODIFICATION ------- #
    def reset(self):
        """
        Reset the tree
        :return:
        """
        self.prune_tree(self)
        self.root = BeliefNode.BeliefNode(self.solver, None, None)
        return self.root

    def reset_root_data(self):
        """
        Completely resets the root data
        :return:
        """
        self.root.data = self.solver.model.create_root_historical_data(self.solver)

    def reset_data(self, root_data=None):
        """
        Keeps information from the provided root node
        :return:
        """
        if root_data is not None:
            self.root.data.reset(root_data)
        else:
            self.root.data.reset()

    def initialize(self):
        self.reset_root_data()
        self.root.action_map = self.solver.action_pool.create_action_mapping(self.root)


