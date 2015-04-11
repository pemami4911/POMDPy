__author__ = 'patrickemami'

import abc
import Point as Pt

class DiscreteAction(Pt.Point):

    def hash(self):
        return self.get_bin_number()

    def equals(self, other_discrete_action):
        assert isinstance(other_discrete_action, DiscreteAction)
        return self.get_bin_number() == other_discrete_action.get_bin_number()

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
        :return:
        """

    @abc.abstractmethod
    def get_bin_number(self):
        """
        Returns the bin number associated with this action
        :return:
        """

    def distance_to(self, other_discrete_action):
        pass
