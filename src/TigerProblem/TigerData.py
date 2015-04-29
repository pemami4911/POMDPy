__author__ = 'patrickemami'

import HistoricalData as Hd
from TigerAction import ActionType
import numpy as np

class TigerData(Hd.HistoricalData):
    """
    Used to store the probabilities that the tiger is behind a certain door.
    This is the belief distribution over the set of possible states.
    For a 2-door system, you have
        P( X = 0 ) = p
        P( X = 1 ) = 1 - p
    """
    def __init__(self):
        self.listen_count = 0
        ''' Initially there is an equal probability of the tiger being in either door'''
        self.door_probabilities = [0.5, 0.5]
        self.legal_actions = self.generate_legal_actions

    def copy(self):
        dat = TigerData()
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
            probability_correct = 0.85
            probability_incorrect = 1.0 - probability_correct
            p1_prior = self.door_probabilities[0]
            p2_prior = self.door_probabilities[1]

            # Observation 1 - the roar came from door 0
            if observation.source_of_roar[0]:
                observation_probability = (probability_correct * p1_prior) + (probability_incorrect * p2_prior)
                p1_posterior = (probability_correct * p1_prior)/observation_probability
                p2_posterior = (probability_incorrect * p2_prior)/observation_probability
            # Observation 2 - the roar came from door 1
            else:
                observation_probability = (probability_incorrect * p1_prior) + (probability_correct * p2_prior)
                p1_posterior = probability_incorrect * p1_prior / observation_probability
                p2_posterior = probability_correct * p2_prior / observation_probability

            next_data.door_probabilities = [p1_posterior, p2_posterior]
        return next_data

    def generate_legal_actions(self):
        '''
        At each non-terminal state, the agent can listen or choose to open the door based on the current door probabilities
        :return:
        '''
        bins = [ActionType.LISTEN]

        if np.random.uniform(0, 1) <= self.door_probabilities[0]:
            bins = bins + [ActionType.OPEN_DOOR_2]
        else:
            bins = bins + [ActionType.OPEN_DOOR_1]

        return bins
