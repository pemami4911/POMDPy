__author__ = 'patrickemami'

import DiscreteAction as Da

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

class RockAction(Da.DiscreteAction):
    """
    Accepted constructors - either RockAction(ActionType) or RockAction(code)
    The Rock sample problem Action class
    -implements get_bin_number() from DiscretizedAction class

    class methods: action_type (Enum) and rock_no
    """

    def __init__(self, action_type):
        self.action_type = action_type
        if self.action_type >= ActionType.CHECK:
            self.rock_no = self.action_type - ActionType.CHECK
        else:
            self.rock_no = 0


    # Override
    def copy(self):
        return RockAction(self.action_type)

    def print_action(self):
        if self.action_type >= ActionType.CHECK:
            print "CHECK"
        elif self.action_type is ActionType.NORTH:
            print "NORTH"
        elif self.action_type is ActionType.EAST:
            print "EAST"
        elif self.action_type is ActionType.SOUTH:
            print "SOUTH"
        elif self.action_type is ActionType.WEST:
            print "WEST"
        elif self.action_type is ActionType.SAMPLE:
            print "SAMPLE"
        else:
            print "UNDEFINED ACTION"

    # Override
    def get_bin_number(self):
        return self.action_type