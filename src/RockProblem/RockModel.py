__author__ = 'patrickemami'

import Model
import config_parser
import GridPosition as Gp
import RockState as Rs
import numpy as np
import RockAction as Ra
import RockObservation as Ro
import RockActionPool as Rap
import RockPositionHistory as Rph
import logging
import json

class RSCellType(object):
    """
    Rocks are enumerated 0, 1, 2, ...
    other cell types should be negative.
    """
    ROCK = 0
    EMPTY = -1
    GOAL = -2
    OBSTACLE = -3

class RockModel(Model.Model):

    def __init__(self, problem_name):
        super(RockModel, self).__init__(problem_name)
        # logging utility
        self.logger = logging.getLogger('Model.RockModel')
        self.config = json.load(open(config_parser.cfg_file, "r"))

        # -------- Model configurations -------- #

        # The reward for sampling a good rock
        self.good_rock_reward = self.config['good_rock_reward']
        # The penalty for sampling a bad rock.
        self.bad_rock_penalty = self.config['bad_rock_penalty']
        # The reward for exiting the map
        self.exit_reward = self.config['exit_reward']
        # The penalty for an illegal move.
        self.illegal_move_penalty = self.config['illegal_move_penalty']
        # penalty for finishing without sampling a rock
        self.finishing_empty_handed = self.config['finishing_empty_handed']
        self.half_efficiency_distance = self.config['half_efficiency_distance']
        self.preferred_action = self.config["preferred_action"]
        self.checking_penalty = self.config["checking_penalty"]

        # ------------- Flags --------------- #
        # Flag that checks whether the agent has yet to successfully sample a rock
        self.sampled_rock_yet = False
        # Flag that keeps track of whether the agent currently believes there are still good rocks out there
        self.any_good_rocks = False

        # ------------- Data Collection ---------- #
        self.unique_rocks_sampled = {}
        self.num_times_sampled = 0
        self.num_reused_nodes = 0

        # The number of rows in the map.
        self.n_rows = 0
        # The number of columns in the map
        self.n_cols = 0
        # The number of rocks on the map.
        self.n_rocks = 0

        self.start_position = Gp.GridPosition()

        # The coordinates of the rocks.
        self.rock_positions = []
        # The coordinates of the goal squares.
        self.goal_positions = []
        # The environment map in vector form.
        # List of lists of RSCellTypes
        self.env_map = []

        # The distance from each cell to the nearest goal square.
        self.goal_distances = []
        # The distance from each cell to each rock.
        self.rock_distances = []

        # Smart rock data
        self.all_rock_data = []

        # Actual rock states
        self.actual_rock_states  = []

        # The environment map in text form.
        self.map_text, dimensions = config_parser.parse_map(self.config['map_file'])
        self.n_rows = int(dimensions[0])
        self.n_cols = int(dimensions[1])
        self.initialize()

        self.min_val = -self.illegal_move_penalty / (1 - self.config['discount_factor'])
        self.max_val = self.good_rock_reward * self.n_rocks + self.exit_reward

    # initialize the maps of the grid
    def initialize(self):
        p = Gp.GridPosition()
        for p.i in range(0, self.n_rows):
            tmp = []
            for p.j in range(0, self.n_cols):
                c = self.map_text[p.i][p.j]

                # initialized to empty
                cell_type = RSCellType.EMPTY

                if c is 'o':
                    self.rock_positions.append(p.copy())
                    cell_type = RSCellType.ROCK + self.n_rocks
                    self.n_rocks += 1
                elif c is 'G':
                    cell_type = RSCellType.GOAL
                    self.goal_positions.append(p.copy())
                elif c is 'S':
                    self.start_position = p.copy()
                    cell_type = RSCellType.EMPTY
                elif c is 'X':
                    cell_type = RSCellType.OBSTACLE
                tmp.append(cell_type)

            self.env_map.append(tmp)

    ''' Utility functions '''
    # returns the RSCellType at the specified position
    def get_cell_type(self, pos):
        assert isinstance(pos, Gp.GridPosition)
        return self.env_map[pos.i][pos.j]

    def get_sensor_correctness_probability(self, distance):
        assert self.half_efficiency_distance is not 0, self.logger.warning("Tried to divide by 0! Naughty naughty!")
        return (1 + np.power(2.0, -distance / self.half_efficiency_distance)) * 0.5

    ''' Sampling '''
    def sample_an_init_state(self):
        self.sampled_rock_yet = False
        self.actual_rock_states = self.sample_rocks()
        return Rs.RockState(self.start_position, self.sample_rocks())

    def sample_state_uninformed(self):
        while True:
            pos = self.sample_position()
            if self.get_cell_type(pos) is not RSCellType.OBSTACLE:
                return Rs.RockState(pos, self.sample_rocks())
        return None

    def sample_position(self):
        i = np.random.random_integers(0, self.n_rows - 1)
        j = np.random.random_integers(0, self.n_cols - 1)
        return Gp.GridPosition(i, j)

    def sample_rocks(self):
        return self.decode_rocks(np.random.random_integers(0, (1 << self.n_rocks) - 1))

    def decode_rocks(self, value):
        rock_states = []
        for i in range(0, self.n_rocks):
            rock_states.append(value & (1 << i))
        return rock_states

    def encode_rocks(self, rock_states):
        value = 0
        for i in range(0, self.n_rocks):
            if rock_states[i]:
                value += (1 << i)
        return value

    ''' ---------------Implementation of abstract Model class ---------------'''
    def is_terminal(self, rock_state):
        assert isinstance(rock_state, Rs.RockState)
        return self.get_cell_type(rock_state.position) is RSCellType.GOAL

    def is_valid(self, pos):
        assert isinstance(pos, Gp.GridPosition)
        return 0 <= pos.i < self.n_rows and 0 <= pos.j < self.n_cols and \
            self.get_cell_type(pos) is not RSCellType.OBSTACLE

    ''' --------------- get next position and state ---------------'''
    def make_adjacent_position(self, pos, action_type):
        assert isinstance(pos, Gp.GridPosition)

        if action_type is Ra.ActionType.NORTH:
            pos.i -= 1
        elif action_type is Ra.ActionType.EAST:
            pos.j += 1
        elif action_type is Ra.ActionType.SOUTH:
            pos.i += 1
        elif action_type is Ra.ActionType.WEST:
            pos.j -= 1
        return pos

    def make_next_position(self, pos, action_type):
        assert isinstance(pos, Gp.GridPosition)

        is_legal = True

        if action_type >= Ra.ActionType.CHECK:
            pass

        elif action_type is Ra.ActionType.SAMPLE:
            # if you took an illegal action and are in an invalid position
            # sampling is not a legal action to take
            if not self.is_valid(pos):
                is_legal = False
            else:
                rock_no = self.get_cell_type(pos)
                if 0 > rock_no or rock_no >= self.n_rocks:
                    #self.logger.warning("Tried to sample a non-existent rock... naughty naughty!")
                    is_legal = False
        else:
            old_position = pos.copy()
            pos = self.make_adjacent_position(pos, action_type)
            if not self.is_valid(pos):
                pos = old_position
                is_legal = False
        return pos, is_legal

    ''' --------------- Black Box Dynamics stuff --------------'''
    def make_next_state(self, state, action):
        action_type = action.action_type
        next_position, is_legal = self.make_next_position(state.position.copy(), action_type)

        if not is_legal:
            # returns a copy of the current state
            return state.copy(), False

        next_state_rock_states = list(state.rock_states)

        # update the any_good_rocks flag
        self.any_good_rocks = False
        for rock in next_state_rock_states:
            if rock:
                self.any_good_rocks = True

        if action_type == Ra.ActionType.SAMPLE:
            self.num_times_sampled += 1

            rock_no = self.get_cell_type(next_position)
            next_state_rock_states[rock_no] = False

        return Rs.RockState(next_position, next_state_rock_states), True

    def make_observation(self, action, next_state):
        assert isinstance(action, Ra.RockAction)
        assert isinstance(next_state, Rs.RockState)

        # generate new observation if not checking or sampling a rock
        if action.action_type < Ra.ActionType.SAMPLE:
            # Defaults to empty cell and Bad Rock
            obs = Ro.RockObservation()
            # self.logger.info("Created Rock Observation - is_good: %s", str(obs.is_good))
            return obs
        elif action.action_type == Ra.ActionType.SAMPLE:
            # The cell is not empty since it contains a rock, and the rock is now "Bad"
            obs = Ro.RockObservation(False, False)
            return obs

        observation = self.actual_rock_states[action.rock_no]

        # if checking a rock...
        dist = next_state.position.euclidean_distance(self.rock_positions[action.rock_no])

        # NOISY OBSERVATION
        # bernoulli distribution is a binomial distribution with n = 1
        # if half efficiency distance is 20, and distance to rock is 20, correct has a 50/50
        # chance of being True. If distance is 0, correct has a 100% chance of being True.
        correct = np.random.binomial(1.0, self.get_sensor_correctness_probability(dist))

        if not correct:
            # Return the incorrect state if the sensors malfunctioned
            observation = not observation

        # If I now believe that a rock, previously bad, is now good, change that here
        if observation and not next_state.rock_states[action.rock_no]:
            next_state.rock_states[action.rock_no] = True
        # Likewise, if I now believe a rock, previously good, is now bad, change that here
        elif not observation and next_state.rock_states[action.rock_no]:
            next_state.rock_states[action.rock_no] = False

        # Normalize the observation
        if observation > 1:
            observation = True

        return Ro.RockObservation(observation, False)

    def make_reward(self, state, action, next_state, is_legal):
        assert isinstance(action, Ra.RockAction)
        assert isinstance(state, Rs.RockState)
        assert isinstance(next_state, Rs.RockState)

        if not is_legal:
            return -self.illegal_move_penalty
        if self.is_terminal(next_state):

            # if the agent never sampled any rocks and yet the agent believes there are still good rocks out there,
            # penalize the agent for trying to escape to the exit !!!
            if not self.sampled_rock_yet and self.any_good_rocks:
                return -self.finishing_empty_handed
            else:
                return self.exit_reward

        if action.action_type is Ra.ActionType.SAMPLE:

            pos = state.position.copy()

            rock_no = self.get_cell_type(pos)
            if 0 <= rock_no < self.n_rocks:

                ''' statistics stuff'''
                # Tried to sample a rock - GOOD
                self.sampled_rock_yet = True

                if not rock_no in self.unique_rocks_sampled:
                        self.unique_rocks_sampled.__setitem__(rock_no, rock_no)
                ''' end statistics stuff '''

                # If the rock ACTUALLY is good, AND I currently believe it to be good, I get rewarded
                if self.actual_rock_states[rock_no] and state.rock_states[rock_no]:
                    # IMPORTANT - After sampling, the rock is marked as
                    # bad to show that it is has been dealt with
                    # "next states".rock_states[rock_no] is set to False in make_next_state
                    state.rock_states[rock_no] = False
                    return self.good_rock_reward
                # otherwise, I either sampled a bad rock I thought was good, sampled a good rock I thought was bad,
                # or sampled a bad rock I thought was bad. All bad behavior!!!
                else:
                    return -self.bad_rock_penalty
            else:
                self.logger.warning("Invalid sample action on non-existent rock while making reward!")
                return -self.illegal_move_penalty

        if action.action_type >= Ra.ActionType.CHECK:
            if state.position.euclidean_distance(self.rock_positions[action.rock_no]) > self.half_efficiency_distance:
                return -self.checking_penalty
            else:
                # reward for checking close to rocks
                return 3 * self.checking_penalty

        return 0

    def generate_reward(self, state, action):
        assert isinstance(action, Ra.RockAction)
        assert isinstance(state, Rs.RockState)

        next_state, is_legal = self.make_next_state(state, action)
        return self.make_reward(state, action, next_state, is_legal)

    def generate_step(self, state, action):
        if action is None:
            self.logger.warning("Tried to generate a step with a null action")
            return None

        assert isinstance(action, Ra.RockAction)
        assert isinstance(state, Rs.RockState)

        result = Model.StepResult()
        result.next_state, is_legal = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state)
        result.reward = self.make_reward(state, action, result.next_state, is_legal)
        result.is_terminal = self.is_terminal(result.next_state)

        return result, is_legal

    ''' ------------ particle generation -----------------'''
    def generate_particles(self, previous_belief, action, obs, n_particles, prev_particles):
        assert isinstance(action, Ra.RockAction)
        assert isinstance(obs, Ro.RockObservation)

        new_particles = []
        if action.action_type >= Ra.ActionType.CHECK:
            # rock no for the rock being Checked
            rock_no = action.rock_no
            weight_map = {}
            weight_total = 0
            for state in prev_particles:
                # get the distance from a previous state particle to this new rock
                dist = state.position.euclidean_distance_to(self.rock_positions[rock_no])
                # whether the previous state particle thought this rock was good or not
                rock_is_good = state.rock_states[rock_no]
                # correctness probability
                probability = self.get_sensor_correctness_probability(dist)
                # if my current observation about this rock does not agree with my previous belief about this rock
                # - take the complement of the probability
                if rock_is_good is not obs.is_good:
                    probability = 1 - probability

                # update the weight with the probability
                if state in weight_map:
                    weight_map[state] += probability
                else:
                    weight_map.__setitem__(state, probability)
                weight_total += probability

                scale = n_particles / weight_total
                for key, value in dict.iteritems(weight_map):
                    proportion = value * scale
                    num_to_add = long(proportion)
                    if np.random.binomial(1.0, proportion - num_to_add):
                        num_to_add += 1
                    for i in range(0, num_to_add):
                        new_particles.append(key.copy())

        # if not a CHECK action, we just add each resultant state
        else:
            for state in prev_particles:
                new_particles.append(self.make_next_state(state, action)[0])
        return new_particles

    def generate_particles_uninformed(self, previous_belief, action, obs, n_particles):
        assert isinstance(action, Ra.RockAction)
        assert isinstance(obs, Ro.RockObservation)

        old_pos = previous_belief.get_states()[0].position

        particles = []
        while particles.__len__() < n_particles:
            old_state = Rs.RockState(old_pos, self.sample_rocks())
            result, is_legal = self.generate_step(old_state, action)
            if obs.equals(result.observation):
                particles.append(result.next_state)
        return particles

    ''' --------------- pretty printing methods --------------- '''
    def disp_cell(self, rs_cell_type):
        if rs_cell_type >= RSCellType.ROCK:
            print hex(rs_cell_type - RSCellType.ROCK),
            return

        if rs_cell_type is RSCellType.EMPTY:
            print '.',
        elif rs_cell_type is RSCellType.GOAL:
            print 'G',
        elif rs_cell_type is RSCellType.OBSTACLE:
            print 'X',
        else:
            print 'ERROR-',
            print rs_cell_type,

    def draw_env(self):
        for row in self.env_map:
            map(self.disp_cell, row)
            print '\n'

    ''' --------------- model customizations --------------- '''
    # Generate a random legal action to take
    def get_legal_action(self, position_data):
        actions = position_data.generate_legal_actions()
        return actions[np.random.random_integers(0, actions.__len__() - 1)]

    def get_all_actions_in_order(self):
        all_actions = []
        for code in range(0, 5 + self.n_rocks):
            all_actions.append(Ra.RockAction(code))
        return all_actions

    def get_random_action(self):
        return Ra.RockAction(np.random.random_integers(0, 4 + self.n_rocks))

    def create_action_pool(self):
        return Rap.RockActionPool(self)

    def create_root_historical_data(self, solver):
        self.create_new_rock_data()
        return Rph.PositionAndRockData(self, self.start_position.copy(), self.all_rock_data, solver)

    def create_new_rock_data(self):
        self.all_rock_data = []
        for i in range(0, self.n_rocks):
            self.all_rock_data.append(Rph.RockData())

    def get_all_observations_in_order(self):
        return [Ro.RockObservation(False, True), Ro.RockObservation(True, False), Ro.RockObservation(False, False)]

    def create_observation_pool(self, solver):
        return super(RockModel, self).create_observation_pool(solver)





