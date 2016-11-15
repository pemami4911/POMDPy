from pomdpy.pomdp import ActionPool
from pomdpy.discrete_pomdp import DiscreteActionMapping
import numpy as np


class DiscreteActionPool(ActionPool):
    """
     * An abstract implementation of the ActionPool interface that considers actions in terms of
     * discrete bins.
     *
    """
    def __init__(self, model):
        """
        :param model:
        """
        self.all_actions = model.get_all_actions()

    def create_action_mapping(self, belief_node):
        return DiscreteActionMapping(belief_node, self, self.create_bin_sequence(belief_node))

    def sample_an_action(self, bin_number):
        return self.all_actions[bin_number]

    def sample_random_action(self):
        return np.random.choice(self.all_actions)

    @staticmethod
    def create_bin_sequence(belief_node):
        """
        Default behavior is to make available only the legal actions for each action node
        :param belief_node:
        :return:
        """
        return belief_node.data.legal_actions()
