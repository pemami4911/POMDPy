__author__ = 'patrickemami'

import abc
from discrete_POMDP import DiscreteObservationMap


class ObservationPool(object):
    """
    Defines the ObservationPool interface, which allows customization of how the mapping for each
    * individual action node is set up.
    *
    * Using a single class in this way allows certain aspects of the mappings to be stored globally,
    * e.g. to keep statistics that are shared across all of the mappings rather than stored on
    * a per-mapping basis.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_observation_mapping(self, action_node):
        """
        Creates an observation mapping for the given action node.
        :param action_node:
        :return: ObservationMapping
        """


class DiscreteObservationPool(ObservationPool):
    """
    An implementation of the ObservationPool interface that is based on a discrete observation
    * space.
    *
    * All of the information is stored inside the individual mapping classes, so this class serves as
    * a simple factory for instances of DiscreteObservationMap.
    """

    def __init__(self, solver):
        self.solver = solver

    def create_observation_mapping(self, action_node):
        return DiscreteObservationMap(action_node, self.solver)
