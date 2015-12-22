__author__ = 'patrickemami'

import numpy as np

from pomdpy.pomdp import model
from tiger_action import *
from tiger_state import TigerState
from tiger_observation import TigerObservation
from tiger_data import TigerData
from pomdpy.discrete_pomdp import DiscreteActionPool


class TigerModel(model.Model):
    def __init__(self, problem_name="Tiger Problem", num_doors=2):
        super(TigerModel, self).__init__(problem_name)
        self.tiger_door = None
        self.num_doors = num_doors
        self.num_states = num_doors
        self.set_init()

    def set_init(self):
        self.tiger_door = np.random.randint(0, self.num_doors)
        print 'The tiger is behind door ' + str(self.tiger_door + 1)

    ''' --------- Abstract Methods --------- '''

    def is_terminal(self, state):
        if state.door_open:
            return True
        else:
            return False

    def sample_an_init_state(self):
        return self.sample_state_uninformed()

    # TODO test
    def sample_state_uninformed(self):
        random_configuration = [0, 1]
        if np.random.uniform(0, 1) <= 0.5:
            random_configuration.reverse()
        return TigerState(False, random_configuration)

    def get_all_states(self):
        """
        Door is closed + either tiger is believed to be behind door 0 or door 1
        :return:
        """
        return [[False, 0, 1], [False, 1, 0]], self.num_states

    def get_all_actions(self):
        """
        Three unique actions
        :return:
        """
        return [TigerAction(ActionType.LISTEN), TigerAction(ActionType.OPEN_DOOR_1),
                TigerAction(ActionType.OPEN_DOOR_2)], 3

    def get_all_observations(self):
        """
        Either the roar of the tiger is heard coming from door 0 or door 1
        :return:
        """
        return [0, 1], 2

    def get_legal_actions(self, _):
        return self.get_all_actions()

    def is_valid(self, _):
        return True

    def reset_for_simulation(self):
        pass

    def reset_for_run(self):
        pass

    def update(self, sim_data):
        pass

    def get_max_undiscounted_return(self):
        return 10

    ''' Factory methods '''

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, agent):
        return TigerData()

    ''' --------- BLACK BOX GENERATION --------- '''

    def generate_step(self, state, action):
        if action is None:
            return None

        result = model.StepResult()
        result.next_state, is_legal = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state)
        result.reward = self.make_reward(action, result.next_state)
        result.is_terminal = self.is_terminal(result.next_state)

        return result, is_legal

    def make_next_state(self, state, action):
        if action.bin_number == ActionType.LISTEN:
            return state, True

        if action.bin_number > 0:
            return TigerState(True, state.door_prizes), True
        else:
            print "make_next_state - Illegal action was used"
        return None, False

    def make_reward(self, action, next_state):
        """
        :param state:
        :param action:
        :param next_state:
        :return: reward
        """

        if action.bin_number == ActionType.LISTEN:
            return -1

        if self.is_terminal(next_state):
            assert action.bin_number > 0
            if action.bin_number - self.num_doors + 1 == self.tiger_door:
                ''' You chose the door with the tiger :( '''
                return -20
            else:
                ''' You chose the door with the reward! '''
                return +10
        else:
            print "make_reward - Illegal action was used"
            return 0

    def make_observation(self, action, next_state):
        '''
        :param action:
        :param next_state:
        :return:
        '''
        if action.bin_number > 0:
            '''
            No new information is gained by opening a door
            Since this action leads to a terminal state, we don't care
            about the observation
            '''
            return TigerObservation(None)
        else:
            obs = ([0, 1], [1, 0])[self.tiger_door == 0]
            probability_correct = np.random.uniform(0, 1)
            if probability_correct <= 0.85:
                next_state.door_prizes = list(obs)
                next_state.door_prizes.reverse()
                return TigerObservation(obs)
            else:
                next_state.door_prizes = list(obs)
                obs.reverse()
                return TigerObservation(obs)

