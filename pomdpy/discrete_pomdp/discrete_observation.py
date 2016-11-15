import abc
from pomdpy.pomdp import Point


class DiscreteObservation(Point):

    def __init__(self, bin_number):
        self.bin_number = bin_number

    def __hash__(self):
        return self.bin_number

    def __eq__(self, other_discrete_observation):
        return self.bin_number == other_discrete_observation.bin_number

    @abc.abstractmethod
    def copy(self):
        """
        :return:
        """

    @abc.abstractmethod
    def distance_to(self, other_discrete_observation):
        """
        :param other_discrete_observation:
        Problem specific distance metric between observations
        :return:
        """

    @abc.abstractmethod
    def print_observation(self):
        """
        pretty printing
        :return:
        """

    @abc.abstractmethod
    def to_string(self):
        """
        Returns a String version of the observation
        :return:
        """