__author__ = 'patrickemami'

import abc

class Point(object):
    """
    Interface for a point-set topology. Each point is an element of a set of points
    i.e. the set of all actions for a given POMDP
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def copy(self):
        """
        :return:
        """

    @abc.abstractmethod
    def hash(self):
        """
        Returns a hash value for this Action, should be consistent with 'equals'
        :return:
        """

    @abc.abstractmethod
    def equals(self, other_point):
        """
        Returns true iff the action is equal to the other action
        :return:
        """

    @abc.abstractmethod
    def distance_to(self, other_point):
        """
        Returns the distance from this point to another point
        :return:
        """