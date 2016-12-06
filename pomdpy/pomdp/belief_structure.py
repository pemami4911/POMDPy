from builtins import object
from future.utils import with_metaclass
import abc


class BeliefStructure(with_metaclass(abc.ABCMeta, object)):

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