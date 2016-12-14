from __future__ import print_function
from builtins import object
from pomdpy.discrete_pomdp import DiscreteAction


class ActionType(object):
    """
    Enumerates the potential TigerActions
    """
    LISTEN = 0
    OPEN_DOOR_1 = 1
    OPEN_DOOR_2 = 2


class TigerAction(DiscreteAction):
    def __init__(self, action_type):
        super(TigerAction, self).__init__(action_type)
        self.bin_number = action_type

    def copy(self):
        return TigerAction(self.bin_number)

    def to_string(self):
        if self.bin_number is ActionType.LISTEN:
            action = "Listening"
        elif self.bin_number is ActionType.OPEN_DOOR_1:
            action = "Opening door 1"
        elif self.bin_number is ActionType.OPEN_DOOR_2:
            action = "Opening door 2"
        else:
            action = "Unknown action type"
        return action

    def print_action(self):
        if self.bin_number is ActionType.LISTEN:
            print("Listening")
        elif self.bin_number is ActionType.OPEN_DOOR_1:
            print("Opening door 1")
        elif self.bin_number is ActionType.OPEN_DOOR_2:
            print("Opening door 2")
        else:
            print("Unknown action type")

    def distance_to(self, other_point):
        pass
