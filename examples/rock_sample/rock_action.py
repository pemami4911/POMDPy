from __future__ import print_function
from builtins import object
from pomdpy.discrete_pomdp import DiscreteAction


class ActionType(object):
    """
    Lists the possible actions and attributes an integer code to each for the Rock sample problem
    """
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    SAMPLE = 4
    CHECK = 5


class RockAction(DiscreteAction):
    """
    -The Rock sample problem Action class
    -Wrapper for storing the bin number. Also stores the rock number for checking actions
    -Handles pretty printing
    """

    def __init__(self, bin_number):
        super(RockAction, self).__init__(bin_number)
        if self.bin_number >= ActionType.CHECK:
            self.rock_no = self.bin_number - ActionType.CHECK
        else:
            self.rock_no = 0

    def copy(self):
        return RockAction(self.bin_number)

    def print_action(self):
        if self.bin_number >= ActionType.CHECK:
            print("CHECK")
        elif self.bin_number is ActionType.NORTH:
            print("NORTH")
        elif self.bin_number is ActionType.EAST:
            print("EAST")
        elif self.bin_number is ActionType.SOUTH:
            print("SOUTH")
        elif self.bin_number is ActionType.WEST:
            print("WEST")
        elif self.bin_number is ActionType.SAMPLE:
            print("SAMPLE")
        else:
            print("UNDEFINED ACTION")

    def to_string(self):
        if self.bin_number >= ActionType.CHECK:
            action = "CHECK"
        elif self.bin_number is ActionType.NORTH:
            action = "NORTH"
        elif self.bin_number is ActionType.EAST:
            action = "EAST"
        elif self.bin_number is ActionType.SOUTH:
            action = "SOUTH"
        elif self.bin_number is ActionType.WEST:
            action = "WEST"
        elif self.bin_number is ActionType.SAMPLE:
            action = "SAMPLE"
        else:
            action = "UNDEFINED ACTION"
        return action

    def distance_to(self, other_point):
        pass
