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
    def initialize(self):
        """
        Carry out initializations
        :return:
        """