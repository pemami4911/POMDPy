from __future__ import print_function
from pomdpy.discrete_pomdp import DiscreteObservation


class TigerObservation(DiscreteObservation):
    """
    For num_doors = 2, there is an 85 % of hearing the roaring coming from the tiger door.
    There is a 15 % of hearing the roaring come from the reward door.

    source_of_roar[0] = 0 (door 1)
    source_of_roar[1] = 1 (door 2)
    or vice versa
    """

    def __init__(self, source_of_roar):
        if source_of_roar is not None:
            super(TigerObservation, self).__init__((1, 0)[source_of_roar[0]])
        else:
            super(TigerObservation, self).__init__(-1)
        self.source_of_roar = source_of_roar

    def copy(self):
        return TigerObservation(self.source_of_roar)

    def equals(self, other_observation):
        return self.source_of_roar == other_observation.source_or_roar

    def distance_to(self, other_observation):
        return (1, 0)[self.source_of_roar == other_observation.source_of_roar]

    def hash(self):
        return self.bin_number

    def print_observation(self):
        if self.source_of_roar is None:
            print("No observation from entering a terminal state")
        elif self.source_of_roar[0]:
            print("Roaring is heard coming from door 1")
        else:
            print("Roaring is heard coming from door 2")

    def to_string(self):
        if self.source_of_roar is None:
            obs = "No observation from entering a terminal state"
        elif self.source_of_roar[0]:
            obs = "Roaring is heard coming from door 1"
        else:
            obs = "Roaring is heard coming from door 2"
        return obs



