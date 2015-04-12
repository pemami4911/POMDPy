__author__ = 'patrickemami'

import abc
from random import shuffle
import BeliefNode as Bn
import DiscreteActionMapping as Dam
import numpy as np

class ActionPool(object):
    """
    An interface class which is a factory for creating action mappings.

    Using a central factory instance allows each individual mapping to interface with this single
    instance; this allows shared statistics to be kept.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_action_mapping(self, belief_node):
        """
        :param belief_node:
        :return action_mapping:
        """

class DiscreteActionPool(ActionPool):
    """
     An abstract implementation of the ActionPool interface that considers actions in terms of
     * discrete bins.
     *
     * This implementation does not distinguish between the actions inside any given bin; however,
     * it allows a custom method to be given to sample an action from a single bin. This allows
     * different actions within the same bin to be sampled, although they will not be considered as
     * different from the solver's perspective.
     *
     * A concrete implementation of this abstract class requires implementations for
     * get_number_of_bins() and sample_an_action() in order to define the discrete bins
     *
     * Additionally, the create_bin_sequence() method must be implemented so that the initial set of
     * actions to try, and the order to try them in, will be set.
    """
    @abc.abstractmethod
    def sample_an_action(self, bin_number):
        """
        :param bin_number:
        :return: the DiscreteAction
        """

    @abc.abstractmethod
    def sample_random_action(self):
        """
        :return: action
        """

    @abc.abstractmethod
    def create_bin_sequence(self, belief_node):
        """
        Creates an initial sequence of bins to try for the given belief node. This has two effects:
        * - The getNextActionToTry() method will return these actions in the given order
        * - Any actions *not* included will be marked as "illegal", and will be completely ignored
        *      by UCB unless they are later marked as legal.
        :param belief_node:
        :return:
        """

    def create_action_mapping(self, belief_node):
        return Dam.DiscreteActionMapping(belief_node, self, self.create_bin_sequence())

class EnumeratedActionPool(DiscreteActionPool):
    """
    * Provides a default implementation for the action mapping interfaces in terms of an enumerated
    * set of actions. This is just like the discrete action mapping, but there is only one action
    * in each bin.
    *
    * Indeed, the actual mapping classes are the same as those for discrete actions;
    * the enumerated action case is handled simply by providing implementations for the pure virtual
    * methods of DiscreteActionPool.
    """

    def __init__(self, all_actions):
        """
        :param model:
        :param all_actions: list of discrete actions
        """
        assert type(all_actions) is list
        self.all_actions = all_actions

    def sample_an_action(self, bin_number):
        return self.all_actions[bin_number]

    def sample_random_action(self):
        return self.all_actions[np.random.random_integers(0, self.all_actions.__len__() - 1)]

    def create_bin_sequence(self, belief_node):
        bins = range(0, self.all_actions.__len__())
        shuffle(bins)
        return bins
