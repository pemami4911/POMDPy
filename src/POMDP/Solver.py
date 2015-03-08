__author__ = 'patrickemami'

import abc
import History
import BeliefTree as Bt
import logging

class Solver(object):
    """
    Owns a Model, BeliefTree which is used to represent the policy,
    Histories, which represent the collection of Histories that make up the Belief Tree.

    Core functionality:
        - generatePolicy() - generates a solution to the POMDP
        - replenishChild() - replenishes particles in a child belief based on its parent belief
                            and the action and observation taken to get there
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, model):
        self.model = model
        self.action_pool = None
        self.observation_pool = None
        self.histories = None # History
        self.policy = None  # BeliefTree
        self.search_strategy = None
        self.estimation_strategy = None
        self.logger = logging.getLogger('Model.Solver')

        self.initialize_empty()

    def initialize_empty(self):
        # basic initialization
        self.initialize()

        # reset these
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)

        # Initialize the root node of the embedded policy in the policy tree
        self.policy.initialize_root()

    def initialize(self):
        # init the core data structures
        self.histories = History.Histories()
        self.policy = Bt.BeliefTree(self)
        self.policy.reset()

    @abc.abstractmethod
    def generate_policy(self):
        """
        This is the main solver method, which generates a policy starting from the root

        Each belief node contains a set of associated particles. The ratio of certain particles representing
        a specific HistoryEntry to other particles for a Belief Node contains the transition probabilities of the system.
        Then, a uniform sampling from the set of particles is done to draw the most likely state from that belief.
        certain part
        :return:
        """

    @abc.abstractmethod
    def generate_episodes(self, n_particles, root_node):
        """
        Randomly samples a state and an action from the set of all available states and actions
        and returns a list of length n of random history entries
        """

    @abc.abstractmethod
    def execute(self):
        """
        Traverse the Belief Tree and return the policy
        :return:
        """


