from builtins import object
import abc
from future.utils import with_metaclass


class Solver(with_metaclass(abc.ABCMeta, object)):
    """
    Base class for all solvers
    """

    def __init__(self, agent):
        self.model = agent.model

    @staticmethod
    @abc.abstractmethod
    def reset(agent):
        """
        Return a new instance of a concrete solver class
        :param agent:
        :return:
        """