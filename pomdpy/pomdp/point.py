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
    def distance_to(self, other_point):
        """
        Returns the distance from this point to another point
        :param other_point:
        :return:
        """