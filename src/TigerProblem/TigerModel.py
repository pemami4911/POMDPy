__author__ = 'patrickemami'

import json

import numpy as np
import random
import Model
from TigerAction import TigerAction, ActionType
from TigerState import TigerState
from TigerObservation import TigerObservation
from TigerActionPool import TigerActionPool
import config_parser


class TigerModel(Model.Model):

    def __init__(self, problem_name="Tiger Problem", num_doors=2):
        super(TigerModel, self).__init__(problem_name)
        self.tiger_door = None
        self.num_doors = num_doors
        self.num_states = num_doors
        self.config = json.load(open(config_parser.cfg_file, "r"))

    def set_init(self):
        self.tiger_door = np.random.randint(0, self.num_doors)

    ''' --------- UTIL --------- '''
    def is_terminal(self, state):
        if state.door_open:
            return True
        else:
            return False

    def sample_an_init_state(self):
        pass

    def sample_state_uninformed(self):
        pass

    # ????
    def get_legal_action(self, data):
        return random.choice(self.get_legal_actions())

    def get_legal_actions(self):
        return [ActionType.LISTEN, ActionType.OPEN_DOOR_0, ActionType.OPEN_DOOR_1]

    def is_valid(self, state):
        return True

    ''' --------- BLACK BOX GENERATION --------- '''
    def generate_step(self, state, action):
        if action is None:
            return None

        assert isinstance(action, TigerAction)
        assert isinstance(state, TigerState)

        result = Model.StepResult()
        result.next_state = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(action)
        result.reward = self.make_reward(action, result.next_state)
        result.is_terminal = self.is_terminal(result.next_state)

        return result

    def make_next_state(self, state, action):
        if action.action_type == ActionType.LISTEN:
            return state

        if action.action_type > 0:
             return TigerState(True, state.door_prizes)
        else:
             print "make_next_state - Illegal action was used"

    def make_reward(self, action, next_state):
        """
        :param state:
        :param action:
        :param next_state:
        :return: reward
        """

        if action.action_type == ActionType.LISTEN:
            return -1

        if self.is_terminal(next_state):
            assert action.action_type > 0
            if action.action_type - self.num_doors + 1 == self.tiger_door:
                ''' You chose the door with the tiger :( '''
                return -20
            else:
                ''' You chose the door with the reward! '''
                return +10
        else:
            print "make_reward - Illegal action was used"
            return 0

    def make_observation(self, action):
        '''
        :param action:
        :param next_state:
        :return:
        '''
        if action.action_type > 0:
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
                return TigerObservation(obs)
            else:
                return TigerObservation(obs.reverse())

    ''' Factory methods '''
    def create_action_pool(self):
        return TigerActionPool(self)

