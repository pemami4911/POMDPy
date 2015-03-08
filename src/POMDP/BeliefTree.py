__author__ = 'patrickemami'

import BeliefNode
import logging

class BeliefTree:
    """
    Contains the BeliefTree class, which represents an entire belief tree.
    *
    * Most of the work is done in the individual classes for the mappings and nodes; this class
    * simply owns a root node, and keeps track of a vector of all of the nodes in the entire tree
    * for convenient iteration and serialization.
    """

    def __init__(self, solver):
        self.logger = logging.getLogger('Model.BeliefTree')
        self.solver = solver
        self.root = None
        self.all_nodes = []

    def get_node(self, id):
        node = self.all_nodes[id]
        assert node.id is id, self.logger.warning("ID mismatch in Belief Tree - ID should be %s", str(id))
        return self.all_nodes[id]

    def get_number_of_nodes(self):
        return self.all_nodes.__len__()

    # -------- INTERNAL METHODS ------- #
    def add_node(self, node):
        # Negative ID => add it to the back of the vector.
        if node.id < 0:
            node.id = self.all_nodes.__len__()
        self.all_nodes.append(node)


    def remove_node(self, node):
        id = node.id
        last_node_id = self.all_nodes.__len__() - 1
        assert 0 <= id <= last_node_id, self.logger.warning("Node ID is out of bounds")
        assert self.all_nodes[id] is node, self.logger.warning("Node ID does not match index")

        if id < last_node_id:
            last_node = self.all_nodes[last_node_id]
            last_node.id = id
            self.all_nodes[id] = last_node

        # removes the last item in the list
        self.all_nodes.pop()

    # --------- TREE MODIFICATION ------- #
    def reset(self):
        self.root = None
        self.root = BeliefNode.BeliefNode(self.solver, None, None)
        return self.root

    def reset_root_data(self):
        self.root.data = self.solver.model.create_root_historical_data(self.solver)

    def initialize_root(self):
        self.reset_root_data()
        self.root.action_map = self.solver.action_pool.create_action_mapping(self.root)


