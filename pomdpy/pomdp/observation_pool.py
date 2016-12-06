from builtins import object
from future.utils import with_metaclass
import abc


class ObservationPool(with_metaclass(abc.ABCMeta, object)):
    """
    Defines the ObservationPool interface, which allows customization of how the mapping for each
    * individual action node is set up.
    *
    * Using a single class in this way allows certain aspects of the mappings to be stored globally,
    * e.g. to keep statistics that are shared across all of the mappings rather than stored on
    * a per-mapping basis.
    """

    @abc.abstractmethod
    def create_observation_mapping(self, action_node):
        """
        Creates an observation mapping for the given action node.
        :param action_node:
        :return: ObservationMapping
        """


