__author__ = 'patrickemami'

from POMDP.ActionPool import EnumeratedActionPool
from POMDP.DiscretePOMDP.DiscreteActionMapping import DiscreteActionMapping

class TigerActionPool(EnumeratedActionPool):

    def __init__(self, tiger_model):
        super(TigerActionPool, self).__init__(tiger_model.get_legal_actions())
        self.tiger_model = tiger_model

    def create_bin_sequence(self, belief_node):
        return belief_node.data.legal_actions()

    def create_action_mapping(self, belief_node):
        return TigerActionMap(belief_node, self, self.create_bin_sequence(belief_node))

class TigerActionMap(DiscreteActionMapping):
    def __init__(self, belief_node, pool, bin_sequence):
        super(TigerActionMap, self).__init__(belief_node, pool, bin_sequence)
        #self.pool.add_mapping(belief_node, self)
