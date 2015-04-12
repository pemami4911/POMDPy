__author__ = 'patrickemami'

from random import shuffle

import ActionPool as Ap
import DiscreteActionMapping as Dam
import BeliefNode as Bn
import RockModel
import GridPosition as Gp
import RockAction

class RockActionPool(Ap.EnumeratedActionPool):
    """
    Class methods: model (RockModel), mappings
    """
    def __init__(self, rock_model):
        super(RockActionPool, self).__init__(rock_model.get_all_actions_in_order())
        self.rock_model = rock_model

    def create_bin_sequence(self, belief_node):
        data = belief_node.data     # historical data
        bins = data.legal_actions()
        return bins

    def create_action_mapping(self, belief_node):
        return RockActionMap(belief_node, self, self.create_bin_sequence(belief_node))

    def add_mapping(self, grid_position, disc_action_map):
        pass

    def remove_mapping(self, data, disc_action_map):
        pass

class RockActionMap(Dam.DiscreteActionMapping):
    """
    A custom mapping class that keeps track of which actions are legal or illegal at each belief
    """
    def __init__(self, belief_node, pool, bin_sequence):
        super(RockActionMap, self).__init__(belief_node, pool, bin_sequence)
        # self.pool.add_mapping(belief_node.data.grid_position, self)

    def destroy_mapping(self):
        self.pool.remove_mapping(self.owner.data.grid_position, self)





