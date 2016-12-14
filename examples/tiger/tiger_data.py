from __future__ import absolute_import
from pomdpy.pomdp import HistoricalData
from .tiger_action import ActionType
import numpy as np


class TigerData(HistoricalData):
    """
    Used to store the probabilities that the tiger is behind a certain door.
    This is the belief distribution over the set of possible states.
    For a 2-door system, you have
        P( X = 0 ) = p
        P( X = 1 ) = 1 - p
    """
    def __init__(self, model):
        self.model = model
        self.listen_count = 0
        ''' Initially there is an equal probability of the tiger being in either door'''
        self.door_probabilities = [0.5, 0.5]
        self.legal_actions = self.generate_legal_actions

    def copy(self):
        dat = TigerData(self.model)
        dat.listen_count = self.listen_count
        dat.door_probabilities = self.door_probabilities
        return dat

    def update(self, other_belief):
        self.door_probabilities = other_belief.data.door_probabilities

    def create_child(self, action, observation):
        next_data = self.copy()

        if action.bin_number > 1:
            ''' for open door actions, the belief distribution over possible states isn't changed '''
            return next_data
        else:
            self.listen_count += 1
            '''
            Based on the observation, the door probabilities should change here.
            This is the key update that affects value function
            '''

            ''' ------- Bayes update of belief state -------- '''

            next_data.door_probabilities = self.model.belief_update(np.array([self.door_probabilities]), action,
                                                                    observation)
        return next_data

    @staticmethod
    def generate_legal_actions():
        """
        At each non-terminal state, the agent can listen or choose to open the door based on the current door probabilities
        :return:
        """
        return [ActionType.LISTEN, ActionType.OPEN_DOOR_1, ActionType.OPEN_DOOR_2]

