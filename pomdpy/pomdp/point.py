from builtins import object
import abc
from future.utils import with_metaclass


class Point(with_metaclass(abc.ABCMeta, object)):
    """
    Interface for a point-set topology. Each point is an element of a set of points
    i.e. the set of all actions for a given POMDP
    """

    @abc.abstractmethod
    def copy(self):
        """
        :return:
        """

    @abc.abstractmethod
    def distance_to(self, other_point):
        """
        Returns the distance from this point to another point
        :param other_point:
        :return:
        """