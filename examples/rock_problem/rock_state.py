from __future__ import print_function
from builtins import range
from pomdpy.discrete_pomdp import DiscreteState


class RockState(DiscreteState):
    """
    The state contains the position of the robot, as well as a boolean value for each rock
    representing whether it is good (true => good, false => bad).

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, grid_position, rock_states):
        if rock_states is not None:
            assert rock_states.__len__() is not 0
        self.position = grid_position
        self.rock_states = rock_states  # list

    def distance_to(self, other_rock_state):
        """
        Distance is measured between beliefs by the sum of the num of different rocks
        """
        assert isinstance(other_rock_state, RockState)
        distance = 0
        # distance = self.position.manhattan_distance(other_rock_state.position)
        for i, j in zip(self.rock_states, other_rock_state.rock_states):
            if i != j:
                distance += 1
        return distance

    def __eq__(self, other_rock_state):
        return self.position == other_rock_state.position and self.rock_states is other_rock_state.rock_states

    def copy(self):
        return RockState(self.position, self.rock_states)

    def __hash__(self):
        """
        Returns a decimal value representing the binary state string
        :return:
        """
        return int(self.to_string(), 2)

    def to_string(self):
        state_string = self.position.to_string()
        state_string += " - "

        for i in self.rock_states:
            if i:
                state_string += "1 "
            else:
                state_string += "0 "
        return state_string

    def print_state(self):
        """
        Pretty printing
        :return:
        """
        self.position.print_position()

        print('Good: {', end=' ')
        good_rocks = []
        bad_rocks = []
        for i in range(0, self.rock_states.__len__()):
            if self.rock_states[i]:
                good_rocks.append(i)
            else:
                bad_rocks.append(i)
        for j in good_rocks:
            print(j, end=' ')
        print('}; Bad: {', end=' ')
        for k in bad_rocks:
            print(k, end=' ')
        print('}')

    def as_list(self):
        """
        Returns a list containing the (i,j) grid position boolean values
        representing the boolean rock states (good, bad)
        :return:
        """
        state_list = [self.position.i, self.position.j]
        for i in range(0, self.rock_states.__len__()):
            if self.rock_states[i]:
                state_list.append(True)
            else:
                state_list.append(False)
        return state_list

    def separate_rocks(self):
        """
        Used for the PyGame sim
        :return:
        """
        good_rocks = []
        bad_rocks = []
        for i in range(0, self.rock_states.__len__()):
            if self.rock_states[i]:
                good_rocks.append(i)
            else:
                bad_rocks.append(i)
        return good_rocks, bad_rocks