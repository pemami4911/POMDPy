import abc
import numpy
from pomdpy.pomdp import Point


class DiscreteState(Point):
    """
    An ABC for a discrete representation of a point in a state space
    """

    @abc.abstractmethod
    def copy(self):
        """
        :return:
        """

    @abc.abstractmethod
    def as_list(self):
        """
        Returns this state as a list of values
        :return:
        """

    @abc.abstractmethod
    def print_state(self):
        """
        Pretty prints the state
        """

    @abc.abstractmethod
    def to_string(self):
        """
        Returns a String of the state
        :return: String
        """

    def __eq__(self, other_state_as_list):
        """
        By default simply checks for equivalency between the two state lists
        """
        assert type(other_state_as_list) is list
        this_as_list = self.as_list()
        for i,j in zip(this_as_list, other_state_as_list):
            if i != j:
                return 0
        return 1

    def distance_to(self, other_state_as_list):
        """
        Calculates the Euclidean distance between the two state lists by default
        """
        assert type(other_state_as_list) is list
        this_as_list = self.as_list()
        dist = 0
        for i, j in zip(this_as_list, other_state_as_list):
            dist += numpy.linalg.norm(i-j)
        return dist





