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
        super(TigerAction, self).__init__(action_type)
        self.bin_number = action_type

    def copy(self):
        return TigerAction(self.bin_number)

    def get_bin_number(self):
        return self.bin_number

    def print_action(self):
        if self.bin_number is ActionType.LISTEN:
            print "Listening"
        elif self.bin_number is ActionType.OPEN_DOOR_0:
            print "Opening door 1"
        elif self.bin_number is ActionType.OPEN_DOOR_1:
            print "Opening door 2"
        else:
            print "Unknown action type"


