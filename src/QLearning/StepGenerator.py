__author__ = 'patrickemami'

import abc
import enum

class SearchStatus(enum.Enum):
    """
    Defines a basic enumeration of the possible states the step generator can be in
    """
    UNINITIALIZED = 0,  # Not yet set up - could indicate a failure to meet preliminary conditions.
    INITIAL = 1,    # Ready to go.
    OUT_OF_STEPS = 2,   # The step generator is out of steps
    TERMINATED = 3,
    CLEAN_FINISH = 4,   # The history is finished now (i.e. a terminal state, or a non-terminal state)
    ERROR = 5   # An error occurred during the search



class StepGenerator(object):
    """
    Abstract class for extending History Sequences
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extend_and_backup(self, history_sequence, maximum_depth):
        """
        Extends and backs up the given HistorySequence, up to a terminal state
        """

    @abc.abstractmethod
    def get_step(self, history_entry, state):
        """
        Generates new steps in a history sequence, one step at a time
        :param history_entry:
        :param state:
        :param historical_data:
        :return: Model.StepResult()
        """