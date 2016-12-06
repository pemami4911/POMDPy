from builtins import object
import abc


class HistoricalData(object):
    """
    An abstract base class for history-based heuristic info; each HistoricalData will be owned
    by a single belief node.

    In order to function a HistoricalData must be able to generate a new, derived HistoricalData
    instance for a child belief node, based on the action and observation taken to get there.
    """
    @abc.abstractmethod
    def copy(self):
        """
        :return: HistoricalData
        """

    @abc.abstractmethod
    def create_child(self, action, observation):
        """
        Generates a new child HistoricalData for a new belief node, based on the action taken
        and observation received in going to that child node.
        :param action:
        :param observation:
        :return: HistoricalData
        """

