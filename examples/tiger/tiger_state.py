from __future__ import print_function
from builtins import str
from pomdpy.discrete_pomdp import DiscreteState


class TigerState(DiscreteState):
    """
    Enumerated state for the Tiger POMDP

    Consists of a boolean "door_open" containing info on whether the state is terminal
    or not. Terminal states are reached after a door is opened. "door_open" will be false
    until an "open door" action is taken. This aspect of the state is *clearly* fully observable

    The list "door_prizes" contains 0's for doors that have tigers behind them, and 1's
    for doors that have prizes behind them. This part of the state is obscured. Listening
    actions are necessary to increase confidence in choosing the right door. A single TigerState represents a
    "guess" of the true belief state - which is the probability distribution over all states

    For a 2-door system, either door_prizes[0] = 0 and door_prizes[1] = 1, or
        door_prizes[0] = 1 and door_prizes[1] = 0

        door_open = False, door_prizes[0] = 0, and door_prizes[1] = 1
        door_open = False, door_prizes[0] = 1, and door_prizes[1] = 0

        --------------------------------------------------------------
        Placeholder for showing that the Markov Chain has reached an absorbing state ->

        door_open = True, door_prizes[0] = X, and door_prizes[1] = X
    """
    def __init__(self, door_open, door_prizes):
        self.door_open = door_open  # lists
        self.door_prizes = door_prizes

    def distance_to(self, other_state):
        return self.equals(other_state)

    def copy(self):
        return TigerState(self.door_open, self.door_prizes)

    def equals(self, other_state):
        if self.door_open == other_state.door_open and \
                self.door_prizes == other_state.door_prizes:
            return 1
        else:
            return 0

    def hash(self):
        pass

    def as_list(self):
        """
        Concatenate both lists
        :return:
        """
        return self.door_open + self.door_prizes

    def to_string(self):
        if self.door_open:
            state = 'Door is open'
        else:
            state = 'Door is closed'
        return state + ' (' + str(self.door_prizes[0]) + ', ' + str(self.door_prizes[1]) + ')'

    def print_state(self):
        print(self.to_string())

