__author__ = 'patrickemami'

import abc
import logging
import json
import numpy as np
import ObservationPool as Op
import config_parser
import random

class Model(object):
    """
    * Defines the Model class, which is an abstract class representing a POMDP model to be solved by
    * the solver. The core of the Model class is represented by a "black box" generative model which
    * is used to generate new steps in a simulation.

    * The key methods are:
    * - sampleAnInitState() - samples a new state from the initial belief.
    * - isTerminal() - returns true iff the given state is terminal.
    * - generateStep() - the key method representing the generative model; basically, it does
    *          (s, a) => (o, r, s')
    * - createActionPool() - this defines the way in which actions are mapped out inside the policy
    *      tree
    * - Create the logger for debugging, which all other loggers will be children of
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, problem_name):
        self.problem_name = problem_name
        self.format = "%(asctime)s - %(name)s - %(message)s"
        logging.basicConfig(filename='POMDPy.log', level=logging.DEBUG, format=self.format)
        self.logger = logging.getLogger('Model')
        self.sys_cfg = json.load(open(config_parser.sys_cfg, "r"))

    @abc.abstractmethod
    def reset(self):
        """
        The Simulator (Model) should be reset before each simulation
        :return:
        """

    @abc.abstractmethod
    def generate_step(self, state, action):
        """
        Generates a full StepResult, including the next state, an observation, and the reward
        *
        * For convenience, the action taken is also included in the result, as well as a flag for
        * whether or not the resulting next state is terminal.
        :param state:
        :param action:
        :return: StepResult
        """

    @abc.abstractmethod
    def sample_an_init_state(self):
        """
        Samples an initial state from the initial belief.
        :return: State
        """

    @abc.abstractmethod
    def sample_state_uninformed(self):
        """
        Samples a state from a poorly-informed prior. This is used by the provided default
        implementation of the second generateParticles() method.
        :return:
        """

    @abc.abstractmethod
    def get_legal_action(self, data):
        """
        :param data:
        :return:
        """

    @abc.abstractmethod
    def get_legal_actions(self):
        """
        :return: list of legal actions
        """

    @abc.abstractmethod
    def is_terminal(self, state):
        """
        Returns true iff the given state is terminal.
        :param state:
        :return:
        """

    @abc.abstractmethod
    def is_valid(self, state):
        """
        Returns true iff the given state is valid
        :param state:
        :return:
        """

    @abc.abstractmethod
    def create_action_pool(self):
        """
        :param solver:
        :return:
        """

    @abc.abstractmethod
    def get_random_action(self):
        """
        :return: a random legal action
        """

    def generate_particles(self, previous_belief, action, obs, n_particles, prev_particles):
        """
        Generates new state particles based on the state particles of the previous node,
        * as well as on the action and observation.
        *
        * The default implementation uses rejection sampling, but this can be overridden to provide
        * a more efficient implementation.
        :param previous_belief:
        :param action:
        :param obs:
        :param n_particles:
        :return: list of particles
        """
        particles = []
        obs_map = previous_belief.action_map.get_action_node(action).observation_map
        child_node = obs_map.get_belief(obs)

        while particles.__len__() < n_particles:
            # sample a random particle
            state = random.choice(prev_particles)

            # Now generate a step in the model, and compare the observation to the actual observation.
            # Note that this comparison is done implicitly via the observation mapping, to ensure
            # that approximate observations are treated cleanly.
            result, is_legal = self.generate_step(state, action)
            if obs_map.get_belief(result.observation) is child_node:
                particles.append(result.next_state)
        return particles

    def generate_particles_uninformed(self, previous_belief, action, obs, n_particles):
        """
        Generates new state particles based on uninformed prior
        :param previous_belief:
        :param action:
        :param obs:
        :param n_particles:
        :return:
        """
        particles = []
        obs_map = previous_belief.action_map.get_action_node(action).observation_map
        child_node = obs_map.get_belief(obs)

        while particles.__len__() < n_particles:
            # sample a random particle
            state = self.sample_state_uninformed()

            # Now generate a step in the model, and compare the observation to the actual observation.
            # Note that this comparison is done implicitly via the observation mapping, to ensure
            # that approximate observations are treated cleanly.
            result, is_legal = self.generate_step(state, action)
            if obs_map.get_belief(result.observation) is child_node:
                particles.append(result.next_state)
        return particles

    def create_observation_pool(self, solver):
        return Op.DiscreteObservationPool(solver)


    def create_search_strategy(self, solver):
        pass

    def create_estimation_strategy(self, solver):
        pass

class StepResult:
    """
     Represents the results of a complete step in the model, including the next state,
     * observation, and reward
     *
     * For convenience, this also includes the action taken, and a boolean flag representing
     * whether or not the resulting next state is a terminal state.
    """
    def __init__(self):
        self.action = None
        self.observation = None
        self.reward = 0
        self.next_state = None
        self.is_terminal = 0

    def print_step_result(self):
        print "------- Step Result --------"
        print "Action: ",
        self.action.print_action()
        print "Observation: ",
        self.observation.print_observation()
        print "Reward: ",
        print self.reward
        print "Next state: ",
        self.next_state.print_state()
        print "Is terminal: ",
        print self.is_terminal