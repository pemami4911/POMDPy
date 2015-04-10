__author__ = 'patrickemami'

import HistoricalData as Hd
import RockAction as Ra
import GridPosition as Gp
import logging
import numpy as np

class RockData:
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
        data_as_string = " Check count: " + str(self.check_count) + " Goodness number: " +\
                         str(self.goodness_number) + " Probability that rock is good: " + str(self.chance_good)

        return data_as_string

class PositionAndRockData(Hd.HistoricalData):
    """
    A class to store the robot position associated with a given belief node, as well as
    explicitly calculated probabilities of goodness for each rock.
    """
    def __init__(self, model, grid_position, all_rock_data, solver):
        assert isinstance(grid_position, Gp.GridPosition)
        self.model = model
        self.solver = solver
        self.grid_position = grid_position

        # List of RockData indexed by the rock number
        self.all_rock_data = all_rock_data

        self.logger = logging.getLogger("Model.RockModel.PositionAndRockData")

        # Holds reference to the function for generating legal actions
        self.legal_actions = self.generate_legal_actions

    def copy(self):
        return PositionAndRockData(self.model, self.grid_position.copy(), self.all_rock_data, self.solver)

    def update(self, other_belief):
        self.all_rock_data = other_belief.data.all_rock_data

    def any_good_rocks(self):
        any_good_rocks = False
        for rock_data in self.all_rock_data:
            if rock_data.goodness_number > 0:
                any_good_rocks = True
        return any_good_rocks

    def create_child(self, rock_action, rock_observation):
        if not isinstance(rock_action, Ra.RockAction):
            rock_action = Ra.RockAction(rock_action)

        next_data = self.copy()
        next_position, is_legal = self.model.make_next_position(self.grid_position.copy(), rock_action.action_type)
        next_data.grid_position = next_position
        if not is_legal:
            self.logger.warning("I tried an illegal action!")

        if rock_action.action_type is Ra.ActionType.SAMPLE:
            rock_no = self.model.get_cell_type(self.grid_position)
            next_data.all_rock_data[rock_no].chance_good = 0.0
            next_data.all_rock_data[rock_no].check_count = 10
            next_data.all_rock_data[rock_no].goodness_number = -10

        elif rock_action.action_type >= Ra.ActionType.CHECK:
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

            assert likelihood_good + likelihood_bad is not 0

            rock_data.chance_good = likelihood_good / (likelihood_good + likelihood_bad)

        return next_data

    def generate_legal_actions(self):
        legal_actions = []
        for action in self.model.get_all_actions_in_order():
            next_position, is_legal = self.model.make_next_position(self.grid_position.copy(), action.action_type)
            if is_legal:    # if the action is legal
                legal_actions.append(action.get_bin_number())
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
                smart_actions.append(Ra.ActionType.SAMPLE)
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
            #Once an interesting rock is found, break out of the for loop

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
            smart_actions.append(Ra.ActionType.EAST)
            return smart_actions


        if north_worth_while:
            smart_actions.append(Ra.ActionType.NORTH)
        if south_worth_while:
            smart_actions.append(Ra.ActionType.SOUTH)
        if east_worth_while:
            smart_actions.append(Ra.ActionType.EAST)
        if west_worth_while:
            smart_actions.append(Ra.ActionType.WEST)

        # see which rocks we might want to check
        for i in range(0, n_rocks):
            rock_data = self.all_rock_data[i]
            if rock_data.chance_good != 0.0 and rock_data.chance_good != 1.0 and np.abs(rock_data.goodness_number) < 2:
                smart_actions.append(Ra.ActionType.CHECK + i)

        return smart_actions

    '''
    These methods determine which actions are considered legal vs. illegal
    '''
    def generate_exploratory_actions(self):
        """
        These actions are simply for the agent to construct a representation of the environment and are used exclusively
        early on
        """
        return [Ra.ActionType.NORTH, Ra.ActionType.SOUTH, Ra.ActionType.EAST, Ra.ActionType.WEST]

    # Generate suggested actions for the agent, according to these *fair* rules
    #   1. If the agent is on top of a rock and the rock has been checked, suggest sampling
    #   2. If the agent knows that there is a worthwhile rock out there, it is suggest the check action for that rock
    #   3. Always suggest all actions
    def generate_sampling_actions(self):

        n_rocks = self.model.n_rocks
        # check if we are currently on top of a rock
        rock_no = self.model.get_cell_type(self.grid_position)

        suggested_actions = []

        # if we are on top of a rock, and it has been checked, try sample it
        if 0 <= rock_no < n_rocks:
            rock_data = self.all_rock_data[rock_no]
            if rock_data.chance_good == 1.0 or rock_data.goodness_number > 0:
                suggested_actions.append(Ra.ActionType.SAMPLE)
                return suggested_actions

        suggested_actions = [Ra.ActionType.NORTH, Ra.ActionType.SOUTH, Ra.ActionType.EAST, Ra.ActionType.WEST]

        # see which rocks we might want to check
        for i in range(0, n_rocks):
            rock_data = self.all_rock_data[i]
            if rock_data.chance_good != 0.0 and rock_data.chance_good != 1.0 and np.abs(rock_data.goodness_number) < 2:
                suggested_actions.append(Ra.ActionType.CHECK + i)

        return suggested_actions

'''
class PositionData(Hd.HistoricalData):
    """
    Stores position data for each visited grid position
    """
    def __init__(self, model, grid_position, all_rock_data):
        assert isinstance(grid_position, Gp.GridPosition)
        self.model = model
        self.grid_position = grid_position
        self.logger = logging.getLogger('Model.RockModel.RockPositionData')
        self.all_rock_data = all_rock_data

    def copy(self):
        return PositionData(self.model, self.grid_position, self.all_rock_data)

    def create_child(self, rock_action, rock_observation):
        assert isinstance(rock_action, Ra.RockAction)
        assert isinstance(rock_observation, Ro.RockObservation)
        # self.model.make_next_position(pos,action_type) returns next_position, is_legal
        next_position, is_legal = self.model.make_next_position(self.grid_position.copy(), rock_action.action_type)
        if not is_legal:    # test is_legal, should be true
            self.logger.warning("I tried an illegal action!")
        return PositionData(self.model, next_position, self.all_rock_data)   # return new PositionData obj

    def generate_legal_actions(self):
        legal_actions = []
        for action in self.model.get_all_actions_in_order():
            pos_to_check = self.grid_position.copy()
            next_position, is_legal = self.model.make_next_position(pos_to_check, action.action_type)
            if is_legal:    # if the action is legal
                legal_actions.append(action.get_bin_number())
        return legal_actions

    def suggest_actions(self):



    def print_position_data(self):
        print "Position:",
        print self.grid_position.print_position()

    def generate_smart_actions(self):
        smart_actions = []

        n_rocks = self.model.n_rocks

        # check if we are currently on top of a rock
        rock_no = self.model.get_cell_type(self.grid_position)

        # if we are on top of a rock, and it has been checked, sample it
        if 0 <= rock_no < n_rocks:
            rock_data = self.all_rock_data[rock_no]
            if rock_data.chance_good == 1.0 or rock_data.goodness_number > 0:
                # smart_actions.append(self.solver.action_pool.sample_an_action(Ra.ActionType.SAMPLE))
                smart_actions.append(Ra.ActionType.SAMPLE)
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

            #Once an interesting rock is found, break out of the for loop
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
            smart_actions.append(Ra.ActionType.EAST)
            return smart_actions


        if north_worth_while:
            smart_actions.append(Ra.ActionType.NORTH)
        if south_worth_while:
            smart_actions.append(Ra.ActionType.SOUTH)
        if east_worth_while:
            smart_actions.append(Ra.ActionType.EAST)
        if west_worth_while:
            smart_actions.append(Ra.ActionType.WEST)

        # see which rocks we might want to check
        for i in range(0, n_rocks):
            rock_data = self.all_rock_data[i]
            if rock_data.chance_good != 0.0 and rock_data.chance_good != 1.0 and np.abs(rock_data.goodness_number) < 2:
                smart_actions.append(Ra.ActionType.CHECK + i)

        return smart_actions

'''







