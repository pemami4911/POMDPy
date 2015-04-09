__author__ = 'patrickemami'


import DiscreteAction as Da

class ActionType(object):
    """
    Enumerates the potential TigerActions
    """
    LISTEN = 0
    OPEN_DOOR_0 = 1
    OPEN_DOOR_1 = 2

class TigerAction(Da.DiscreteAction):

    def __init__(self, action_type):
        self.action_type = action_type

    def copy(self):
        return TigerAction(self.action_type)

    def get_bin_number(self):
        return self.action_type

    def print_action(self):
        if self.action_type is ActionType.LISTEN:
            print "Listening"
        elif self.action_type is ActionType.OPEN_DOOR_0:
            print "Opening door 1"
        elif self.action_type is ActionType.OPEN_DOOR_1:
            print "Opening door 2"
        else:
            print "Unknown action type"


