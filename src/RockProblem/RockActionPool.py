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
        assert isinstance(rock_model, RockModel.RockModel)
        super(RockActionPool, self).__init__(rock_model.get_all_actions_in_order())
        self.rock_model = rock_model
        self.mappings = {}  # dictionary of GridPosition: (set of) DiscreteActionMaps
        self.preferred_actions = []

        if self.rock_model.preferred_action == "SAMPLE":
            self.preferred_actions.append(RockAction.ActionType.SAMPLE)

    def create_bin_sequence(self, belief_node):
        assert isinstance(belief_node, Bn.BeliefNode)

        data = belief_node.data     # historical data

        # update the current set of actions that are legal / not legal
        #data.update()

        bins = data.legal_actions()

        return bins

    def create_action_mapping(self, belief_node):
        assert isinstance(belief_node, Bn.BeliefNode)
        return RockActionMap(belief_node, self, self.create_bin_sequence(belief_node), self.preferred_actions)

    def add_mapping(self, grid_position, disc_action_map):
        assert isinstance(grid_position, Gp.GridPosition)
        # add an empty set at the grid position in the dictionary if there isn't already one
        if not grid_position in self.mappings:
            self.mappings.__setitem__(grid_position, set())
        self.mappings.get(grid_position).add(disc_action_map)

    def remove_mapping(self, grid_position, disc_action_map):
        if grid_position in self.mappings:
            self.mappings.get(grid_position).remove(disc_action_map)

class RockActionMap(Dam.DiscreteActionMapping):
    """
    A custom mapping class that keeps track of which actions are legal or illegal at each belief
    """
    def __init__(self, belief_node, pool, bin_sequence, preferred_actions):
        super(RockActionMap, self).__init__(belief_node, pool, bin_sequence, preferred_actions)
        self.pool.add_mapping(belief_node.data.grid_position, self)

    def destroy_mapping(self):
        self.pool.remove_mapping(self.owner.data.grid_position, self)





