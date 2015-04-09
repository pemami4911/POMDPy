__author__ = 'patrickemami'

import ActionPool
import DiscreteActionMapping

class TigerActionPool(ActionPool.EnumeratedActionPool):

    def __init__(self, tiger_model):
        super(TigerActionPool, self).__init__(tiger_model.get_legal_actions())
        self.tiger_model = tiger_model
        self.mappings = {}

    def create_bin_sequence(self, belief_node):
        # All possible states have the same set of legal actions, so we don't care about the current belief node data
        return self.tiger_model.get_legal_actions()

    def create_action_mapping(self, belief_node):
        return TigerActionMap(belief_node, self, self.create_bin_sequence(None))

    def add_mapping(self, belief_node, disc_action_map):

        # add an empty set at the grid position in the dictionary if there isn't already one
        if not belief_node in self.mappings:
            self.mappings.__setitem__(belief_node, set())
        self.mappings.get(belief_node).add(disc_action_map)

    def remove_mapping(self, belief_node, disc_action_map):
        if belief_node in self.mappings:
            self.mappings.get(belief_node).remove(disc_action_map)

class TigerActionMap(DiscreteActionMapping.DiscreteActionMapping):
    def __init__(self, belief_node, pool, bin_sequence):
        super(TigerActionMap, self).__init__(belief_node, pool, bin_sequence, None)
        self.pool.add_mapping(belief_node, self)

    def destroy_mapping(self):
        self.pool.remove_mapping(self.owner, self)