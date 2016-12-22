from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
import numpy as np
from pomdpy.pomdp import model
from .tiger_action import *
from .tiger_state import TigerState
from .tiger_observation import TigerObservation
from .tiger_data import TigerData
from pomdpy.discrete_pomdp import DiscreteActionPool
from pomdpy.discrete_pomdp import DiscreteObservationPool


class TigerModel(model.Model):
    def __init__(self, problem_name="Tiger"):
        super(TigerModel, self).__init__(problem_name)
        self.tiger_door = None
        self.num_doors = 2
        self.num_states = 2
        self.num_actions = 3
        self.num_observations = 2

    def start_scenario(self):
        self.tiger_door = np.random.randint(0, self.num_doors) + 1

    ''' --------- Abstract Methods --------- '''

    def is_terminal(self, state):
        if state.door_open:
            return True
        else:
            return False

    def sample_an_init_state(self):
        return self.sample_state_uninformed()

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def sample_state_uninformed(self):
        random_configuration = [0, 1]
        if np.random.uniform(0, 1) <= 0.5:
            random_configuration.reverse()
        return TigerState(False, random_configuration)

    def sample_state_informed(self, belief):
        """

        :param belief:
        :return:
        """
        s = 100. * np.random.random()
        int1 = np.array([0., 100. * belief[0]])
        if int1[0] <= s <= int1[1]:
            return TigerState(False, [0, 1])
        else:
            return TigerState(False, [1, 0])

    def get_all_states(self):
        """
        Door is closed + either tiger is believed to be behind door 0 or door 1
        :return:
        """
        return [[False, 0, 1], [False, 1, 0]]

    def get_all_actions(self):
        """
        Three unique actions
        :return:
        """
        return [TigerAction(ActionType.LISTEN), TigerAction(ActionType.OPEN_DOOR_1),
                TigerAction(ActionType.OPEN_DOOR_2)]

    def get_all_observations(self):
        """
        Either the roar of the tiger is heard coming from door 0 or door 1
        :return:
        """
        return [0, 1]

    def get_legal_actions(self, _):
        return self.get_all_actions()

    def is_valid(self, _):
        return True

    def reset_for_simulation(self):
        self.start_scenario()

    # Reset every "episode"
    def reset_for_epoch(self):
        self.start_scenario()

    def update(self, sim_data):
        pass

    def get_max_undiscounted_return(self):
        return 10

    @staticmethod
    def get_transition_matrix():
        """
        |A| x |S| x |S'| matrix, for tiger problem this is 3 x 2 x 2
        :return:
        """
        return np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]]
        ])

    @staticmethod
    def get_observation_matrix():
        """
        |A| x |S| x |O| matrix
        :return:
        """
        return np.array([
            [[0.85, 0.15], [0.15, 0.85]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]]
        ])

    @staticmethod
    def get_reward_matrix():
        """
        |A| x |S| matrix
        :return:
        """
        return np.array([
            [-1., -1.],
            [-20.0, 10.0],
            [10.0, -20.0]
        ])

    @staticmethod
    def get_initial_belief_state():
        return np.array([0.5, 0.5])

    ''' Factory methods '''

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, agent):
        return TigerData(self)

    ''' --------- BLACK BOX GENERATION --------- '''

    def generate_step(self, action, state=None):
        if action is None:
            print("ERROR: Tried to generate a step with a null action")
            return None
        elif not isinstance(action, TigerAction):
            action = TigerAction(action)

        result = model.StepResult()
        result.is_terminal = self.make_next_state(action)
        result.action = action.copy()
        result.observation = self.make_observation(action)
        result.reward = self.make_reward(action, result.is_terminal)

        return result

    @staticmethod
    def make_next_state(action):
        if action.bin_number == ActionType.LISTEN:
            return False
        else:
            return True

    def make_reward(self, action, is_terminal):
        """
        :param action:
        :param is_terminal:
        :return: reward
        """

        if action.bin_number == ActionType.LISTEN:
            return -1.0

        if is_terminal:
            assert action.bin_number > 0
            if action.bin_number == self.tiger_door:
                ''' You chose the door with the tiger '''
                # return -20
                return -20.
            else:
                ''' You chose the door with the prize! '''
                return 10.0
        else:
            print("make_reward - Illegal action was used")
            return 0.0

    def make_observation(self, action):
        """
        :param action:
        :return:
        """
        if action.bin_number > 0:
            '''
            No new information is gained by opening a door
            Since this action leads to a terminal state, we don't care
            about the observation
            '''
            return TigerObservation(None)
        else:
            obs = ([0, 1], [1, 0])[self.tiger_door == 1]
            probability_correct = np.random.uniform(0, 1)
            if probability_correct <= 0.85:
                return TigerObservation(obs)
            else:
                obs.reverse()
                return TigerObservation(obs)

    def belief_update(self, old_belief, action, observation):
        """
        Belief is a 2-element array, with element in pos 0 signifying probability that the tiger is behind door 1

        :param old_belief:
        :param action:
        :param observation:
        :return:
        """
        if action > 1:
            return old_belief

        probability_correct = 0.85
        probability_incorrect = 1.0 - probability_correct
        p1_prior = old_belief[0]
        p2_prior = old_belief[1]

        # Observation 1 - the roar came from door 0
        if observation.source_of_roar[0]:
            observation_probability = (probability_correct * p1_prior) + (probability_incorrect * p2_prior)
            p1_posterior = old_div((probability_correct * p1_prior),observation_probability)
            p2_posterior = old_div((probability_incorrect * p2_prior),observation_probability)
        # Observation 2 - the roar came from door 1
        else:
            observation_probability = (probability_incorrect * p1_prior) + (probability_correct * p2_prior)
            p1_posterior = probability_incorrect * p1_prior / observation_probability
            p2_posterior = probability_correct * p2_prior / observation_probability
        return np.array([p1_posterior, p2_posterior])
