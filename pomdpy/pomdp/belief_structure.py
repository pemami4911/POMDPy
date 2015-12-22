__author__ = 'patrickemami'

import abc


class BeliefStructure(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        """
        Reset the policy
        :return:
        """

    @abc.abstractmethod
    def initialize(self, init_value=None):
        """
        Carry out initializations of each element of the belief structure,
        setting each element to init_value
        :return:
        """