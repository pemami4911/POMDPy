__author__ = 'patrickemami'

import DiscreteObservation as Do

class TigerObservation(Do.DiscreteObservation):
    """
    For num_doors = 2, there is an 85 % of hearing the roaring coming from the tiger door.
    There is a 15 % of hearing the roaring come from the reward door.

    source_of_roar[0] = 0 (door 0)
    source_of_roar[1] = 1 (door 1)
    or vice versa
    """
    def __init__(self, source_of_roar):
        super(TigerObservation, self).__init__((1, 0)[self.source_of_roar[0]])
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
        if self.source_of_roar[0]:
            print "Roaring is heard coming from door 0"
        else:
            print "Roaring is heard coming from door 1"






