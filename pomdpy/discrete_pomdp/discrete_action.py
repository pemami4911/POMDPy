import abc
from pomdpy.pomdp.point import Point


class DiscreteAction(Point):

    def __init__(self, bin_number):
        self.bin_number = bin_number

    def __hash__(self):
        return self.bin_number

    def __eq__(self, other_discrete_action):
        return self.bin_number == other_discrete_action.bin_number

    @abc.abstractmethod
    def print_action(self):
        """
        Pretty prints the action type
        :return:
        """

    @abc.abstractmethod
    def to_string(self):
        """
        Returns a String version of the action type
        :return:
        """
    @abc.abstractmethod
    def copy(self):
        """
        Returns a proper copy of the Discrete Action
        :return:
        """
