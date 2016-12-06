from builtins import object
from future.utils import with_metaclass
import abc


class ActionPool(with_metaclass(abc.ABCMeta, object)):
    """
    An interface class which is a factory for creating action mappings.

    Using a central factory instance allows each individual mapping to interface with this single
    instance; this allows shared statistics to be kept.
    """

    @abc.abstractmethod
    def create_action_mapping(self, belief_node):
        """
        :param belief_node:
        :return action_mapping:
        """




