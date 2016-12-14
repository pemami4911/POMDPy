from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
from pomdpy.pomdp import HistoricalData
from .rock_action import ActionType
import itertools


# Utility function
class RockData(object):
    """
    Stores data about each rock
    """

    def __init__(self):
        # The number of times this rock has been checked.
        self.check_count = 0
        # The "goodness number"; +1 for each good observation of this rock, and -1 for each bad
        # observation of this rock.
        self.goodness_number = 0
        # The calculated probability that this rock is good.
        self.chance_good = 0.5

    def to_string(self):
        """
        Pretty printing
        """
        data_as_string = " Check count: " + str(self.check_count) + " Goodness number: " + \
                         str(self.goodness_number) + " Probability that rock is good: " + str(self.chance_good)
        return data_as_string


class PositionAndRockData(HistoricalData):
    """
    A class to store the robot position associated with a given belief node, as well as
    explicitly calculated probabilities of goodness for each rock.
    """

    def __init__(self, model, grid_position, all_rock_data, solver):
        self.model = model
        self.solver = solver
        self.grid_position = grid_position

        # List of RockData indexed by the rock number
        self.all_rock_data = all_rock_data

        # Holds reference to the function for generating legal actions
        if self.model.preferred_actions:
            self.legal_actions = self.generate_smart_actions
        else:
            self.legal_actions = self.generate_legal_actions

    @staticmethod
    def copy_rock_data(other_data):
        new_rock_data = []
        [new_rock_data.append(RockData()) for _ in other_data]
        for i, j in zip(other_data, new_rock_data):
            j.check_count = i.check_count
            j.chance_good = i.chance_good
            j.goodness_number = i.goodness_number
        return new_rock_data

    def copy(self):
        """
        Default behavior is to return a shallow copy
        """
        return self.shallow_copy()

    def deep_copy(self):
        """
        Passes along a reference to the rock data to the new copy of RockPositionHistory
        """
        return PositionAndRockData(self.model, self.grid_position.copy(), self.all_rock_data, self.solver)

    def shallow_copy(self):
        """
        Creates a copy of this object's rock data to pass along to the new copy
        """
        new_rock_data = self.copy_rock_data(self.all_rock_data)
        return PositionAndRockData(self.model, self.grid_position.copy(), new_rock_data, self.solver)

    def update(self, other_belief):
        self.all_rock_data = other_belief.data.all_rock_data

    def any_good_rocks(self):
        any_good_rocks = False
        for rock_data in self.all_rock_data:
            if rock_data.goodness_number > 0:
                any_good_rocks = True
        return any_good_rocks

    def create_child(self, rock_action, rock_observation):
        next_data = self.deep_copy()
        next_position, is_legal = self.model.make_next_position(self.grid_position.copy(), rock_action.bin_number)
        next_data.grid_position = next_position

        if rock_action.bin_number is ActionType.SAMPLE:
            rock_no = self.model.get_cell_type(self.grid_position)
            next_data.all_rock_data[rock_no].chance_good = 0.0
            next_data.all_rock_data[rock_no].check_count = 10
            next_data.all_rock_data[rock_no].goodness_number = -10

        elif rock_action.bin_number >= ActionType.CHECK:
            rock_no = rock_action.rock_no
            rock_pos = self.model.rock_positions[rock_no]

            dist = self.grid_position.euclidean_distance(rock_pos)
            probability_correct = self.model.get_sensor_correctness_probability(dist)
            probability_incorrect = 1 - probability_correct

            rock_data = next_data.all_rock_data[rock_no]
            rock_data.check_count += 1

            likelihood_good = rock_data.chance_good
            likelihood_bad = 1 - likelihood_good

            if rock_observation.is_good:
                rock_data.goodness_number += 1
                likelihood_good *= probability_correct
                likelihood_bad *= probability_incorrect
            else:
                rock_data.goodness_number -= 1
                likelihood_good *= probability_incorrect
                likelihood_bad *= probability_correct

            if np.abs(likelihood_good) < 0.01 and np.abs(likelihood_bad) < 0.01:
                # No idea whether good or bad. reset data
                # print "Had to reset RockData"
                rock_data = RockData()
            else:
                rock_data.chance_good = old_div(likelihood_good, (likelihood_good + likelihood_bad))

        return next_data

    def generate_legal_actions(self):
        legal_actions = []
        all_actions = range(0, 5 + self.model.n_rocks)
        new_pos = self.grid_position.copy()
        i = new_pos.i
        j = new_pos.j

        for action in all_actions:
            if action is ActionType.NORTH:
                new_pos.i -= 1
            elif action is ActionType.EAST:
                new_pos.j += 1
            elif action is ActionType.SOUTH:
                new_pos.i += 1
            elif action is ActionType.WEST:
                new_pos.j -= 1

            if not self.model.is_valid_pos(new_pos):
                new_pos.i = i
                new_pos.j = j
                continue
            else:
                if action is ActionType.SAMPLE:
                    rock_no = self.model.get_cell_type(new_pos)
                    if 0 > rock_no or rock_no >= self.model.n_rocks:
                        continue
                new_pos.i = i
                new_pos.j = j
                legal_actions.append(action)
        return legal_actions

    def generate_smart_actions(self):

        smart_actions = []

        n_rocks = self.model.n_rocks

        # check if we are currently on top of a rock
        rock_no = self.model.get_cell_type(self.grid_position)

        # if we are on top of a rock, and it has been checked, sample it
        if 0 <= rock_no < n_rocks:
            rock_data = self.all_rock_data[rock_no]
            if rock_data.chance_good == 1.0 or rock_data.goodness_number > 0:
                smart_actions.append(ActionType.SAMPLE)
                return smart_actions

        worth_while_rock_found = False
        north_worth_while = False
        south_worth_while = False
        east_worth_while = False
        west_worth_while = False

        # Check to see which rocks are worthwhile

        # Only pursue one worthwhile rock at a time to prevent the agent from getting confused and
        # doing nothing
        for i in range(0, n_rocks):
            # Once an interesting rock is found, break out of the for loop

            if worth_while_rock_found:
                break
            rock_data = self.all_rock_data[i]
            if rock_data.chance_good != 0.0 and rock_data.goodness_number >= 0:
                worth_while_rock_found = True
                pos = self.model.rock_positions[i]
                if pos.i > self.grid_position.i:
                    south_worth_while = True
                elif pos.i < self.grid_position.i:
                    north_worth_while = True
                if pos.j > self.grid_position.j:
                    east_worth_while = True
                elif pos.j < self.grid_position.j:
                    west_worth_while = True

        # If no worth while rocks were found, just head east
        if not worth_while_rock_found:
            smart_actions.append(ActionType.EAST)
            return smart_actions

        if north_worth_while:
            smart_actions.append(ActionType.NORTH)
        if south_worth_while:
            smart_actions.append(ActionType.SOUTH)
        if east_worth_while:
            smart_actions.append(ActionType.EAST)
        if west_worth_while:
            smart_actions.append(ActionType.WEST)

        # see which rocks we might want to check
        for i in range(0, n_rocks):
            rock_data = self.all_rock_data[i]
            if rock_data.chance_good != 0.0 and rock_data.chance_good != 1.0 and np.abs(rock_data.goodness_number) < 2:
                smart_actions.append(ActionType.CHECK + i)

        return smart_actions







