__author__ = 'patrickemami'

import abc
import itertools
import numpy
import Point as Pt

'''
An abc for a discrete representation of a point in a state space
'''
class DiscreteState(Pt.Point):

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

    '''
    Need to provide default value
    '''
    @abc.abstractmethod
    def hash(self):
        """
        Returns a hash value for this state, should be consistent with 'equals'
        :return:
        """


    """
    By default simply checks for equivalency between the two lists
    """
    def equals(self, other_state_as_list):
        assert type(other_state_as_list) is list
        this_as_list = self.as_list()
        for i,j in itertools.izip(this_as_list, other_state_as_list):
            if i != j:
                return 0
        return 1

    """
    Calculates the Euclidean distance between the two lists by default
    """
    def distance_to(self, other_state_as_list):
        assert type(other_state_as_list) is list
        this_as_list = self.as_list()
        dist = 0
        for i,j in itertools.izip(this_as_list, other_state_as_list):
            dist += numpy.linalg.norm(i-j)
        return dist





