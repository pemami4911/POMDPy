__author__ = 'patrickemami'

import abc


class Solver(object):
    """
    All POMDP solvers must implement the interface specified below
    Ex. See MCTS (Monte-Carlo Tree Search)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, agent, model):
        self.agent = agent
        self.model = model
        # runner owns Histories, the collection of History Sequences.
        # There is one sequence per run of the MCTS algorithm
        self.history = self.agent.histories.create_sequence()

    @staticmethod
    @abc.abstractmethod
    def reset(runner, model):
        """
        Should return a new instance of a concrete solver class
        :return:
        """

    @abc.abstractmethod
    def select_action(self):
        """
        Call methods specific to the implementation of the solver
        to select an action
        :return:
        """

    @abc.abstractmethod
    def update(self, step_result):
        """
        Feed back the step result, updating the policy representation,
        extending the history, updating particle sets, etc
        :return:
        """

