from __future__ import print_function
from builtins import str
from pomdpy.discrete_pomdp import DiscreteObservation


class RockObservation(DiscreteObservation):
    """
    Default behavior is for the rock observation to say that the rock is empty
    """
    def __init__(self, is_good=False, is_empty=None):
        super(RockObservation, self).__init__(0 if is_empty else (2, 1)[is_good])
        self.is_empty = (True, is_empty)[is_empty is not None]
        self.is_good = is_good

    def distance_to(self, other_rock_observation):
        return abs(self.is_good - other_rock_observation.is_good)

    def copy(self):
        return RockObservation(self.is_good, self.is_empty)

    def __eq__(self, other_rock_observation):
        return self.is_good == other_rock_observation.is_good

    def __hash__(self):
        return (False, True)[self.is_good]

    def print_observation(self):
        if self.is_empty:
            print("EMPTY")
        elif self.is_good == 1:
            print("Good")
        elif self.is_good == 2:
            print("Bad")
        else:
            print(self.is_good)

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
