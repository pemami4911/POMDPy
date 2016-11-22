import abc


class Solver(object):
    """
    Base class for all solvers
    """
    __metaclass__ = abc.ABCMeta

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