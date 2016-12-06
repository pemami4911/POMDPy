from __future__ import absolute_import
from pomdpy.pomdp import ObservationPool
from .discrete_observation_mapping import DiscreteObservationMap


class DiscreteObservationPool(ObservationPool):
    """
    An implementation of the ObservationPool interface that is based on a discrete observation
    space.

    All of the information is stored inside the individual mapping classes, so this class serves as
    a simple factory for instances of DiscreteObservationMap.
    """

    def __init__(self, agent):
        self.agent = agent

    def create_observation_mapping(self, action_node):
        return DiscreteObservationMap(action_node, self.agent)
