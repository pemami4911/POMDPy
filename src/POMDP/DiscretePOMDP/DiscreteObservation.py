__author__ = 'patrickemami'

import abc
import Point as Pt

class DiscreteObservation(Pt.Point):

    def hash(self):
        return self.get_bin_number()

    def equals(self, other_discrete_observation):
        assert isinstance(other_discrete_observation, DiscreteObservation)
        return self.get_bin_number() == other_discrete_observation.get_bin_number()

    @abc.abstractmethod
    def copy(self):
        """
        :return:
        """

    @abc.abstractmethod
    def distance_to(self, other_discrete_observation):
        """
        :param other_discrete_observation:
        :return:
        """

    @abc.abstractmethod
    def get_bin_number(self):
        """
        Returns the bin number associated with this observation
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