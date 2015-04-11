__author__ = 'patrickemami'

import DiscreteObservation as Do


class RockObservation(Do.DiscreteObservation):

    '''
    Default behavior is for the rock observation to say that the rock is empty
    '''
    def __init__(self, is_good=False, is_empty=None):
        self.is_empty = (True, is_empty)[is_empty is not None]
        self.is_good = is_good

    def distance_to(self, other_rock_observation):
        return (1, 0)[self.is_good is other_rock_observation.is_good]

    def copy(self):
        return RockObservation(self.is_good, self.is_empty)

    def equals(self, other_rock_observation):
        return self.is_good == other_rock_observation.is_good

    def hash(self):
        return (False, True)[self.is_good]

    def print_observation(self):
        if self.is_empty:
            print "EMPTY"
        elif self.is_good == 1:
            print "Good"
        elif self.is_good == 2:
            print "Bad"
        else:
            print self.is_good

    def to_string(self):
        if self.is_empty:
            obs = "EMPTY"
        elif self.is_good == 1:
            obs = "Good"
        elif self.is_good == 2:
            obs = "Bad"
        else:
            obs = str(self.is_good)
        return obs

    def get_bin_number(self):
        if self.is_empty:
            return 0
        return (2, 1)[self.is_good]
