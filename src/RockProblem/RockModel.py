__author__ = 'patrickemami'

import logging
import json
import config_parser
from GridPosition import GridPosition
from RockState import RockState
from RockAction import RockAction
from RockObservation import RockObservation
from RockActionPool import RockActionPool
from POMDP.Model import Model, StepResult

# import numpy from RockPositionHistory
from RockPositionHistory import *

class RSCellType(object):
    """
    Rocks are enumerated 0, 1, 2, ...
    other cell types should be negative.
    """
    ROCK = 0
    EMPTY = -1
    GOAL = -2
    OBSTACLE = -3

class RockModel(Model):

    def __init__(self, problem_name):
        super(RockModel, self).__init__(problem_name)
        # logging utility
        self.logger = logging.getLogger('Model.RockModel')
        self.rock_config = json.load(open(config_parser.rock_cfg, "r"))

        # -------- Model configurations -------- #

        # The reward for sampling a good rock
        self.good_rock_reward = self.rock_config['good_rock_reward']
        # The penalty for sampling a bad rock.
        self.bad_rock_penalty = self.rock_config['bad_rock_penalty']
        # The reward for exiting the map
        self.exit_reward = self.rock_config['exit_reward']
        # The penalty for an illegal move.
        self.illegal_move_penalty = self.rock_config['illegal_move_penalty']
        # penalty for finishing without sampling a rock
        self.finishing_empty_handed = self.rock_config['finishing_empty_handed']
        self.half_efficiency_distance = self.rock_config['half_efficiency_distance']

        # ------------- Flags --------------- #
        # Flag that checks whether the agent has yet to successfully sample a rock
        self.sampled_rock_yet = False
        # Flag that keeps track of whether the agent currently believes there are still good rocks out there
        self.any_good_rocks = False

        # ------------- Data Collection ---------- #
        self.unique_rocks_sampled = []
        self.num_times_sampled = 0.0
        self.good_samples = 0.0
        self.num_reused_nodes = 0
        self.num_bad_rocks_sampled = 0
        self.num_good_checks = 0
        self.num_bad_checks = 0

        # -------------- Map data ---------------- #
        # The number of rows in the map.
        self.n_rows = 0
        # The number of columns in the map
        self.n_cols = 0
        # The number of rocks on the map.
        self.n_rocks = 0
        self.num_states = 0
        self.min_val = 0
        self.max_val = 0

        self.start_position = GridPosition()

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
        self.map_text, dimensions = config_parser.parse_map(self.rock_config['map_file'])
        self.n_rows = int(dimensions[0])
        self.n_cols = int(dimensions[1])

        ''' Initialization of map '''
        self.initialize()

    # initialize the maps of the grid
    def initialize(self):
        p = GridPosition()
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
        # Total number of distinct states
        self.num_states = pow(2, self.n_rocks)
        self.min_val = -self.illegal_move_penalty / (1 - self.sys_cfg['discount'])
        self.max_val = self.good_rock_reward * self.n_rocks + self.exit_reward
        self.actual_rock_states = self.sample_rocks()
        print "Actual rock states = ", self.actual_rock_states

    ''' Utility functions '''
    # returns the RSCellType at the specified position
    def get_cell_type(self, pos):
        return self.env_map[pos.i][pos.j]

    def get_sensor_correctness_probability(self, distance):
        assert self.half_efficiency_distance is not 0, self.logger.warning("Tried to divide by 0! Naughty naughty!")
        return (1 + np.power(2.0, -distance / self.half_efficiency_distance)) * 0.5

    ''' Sampling '''
    def sample_an_init_state(self):
        self.sampled_rock_yet = False
        self.unique_rocks_sampled = []
        return RockState(self.start_position, self.sample_rocks())

    def sample_state_uninformed(self):
        while True:
            pos = self.sample_position()
            if self.get_cell_type(pos) is not RSCellType.OBSTACLE:
                return RockState(pos, self.sample_rocks())
        return None

    def sample_position(self):
        i = np.random.random_integers(0, self.n_rows - 1)
        j = np.random.random_integers(0, self.n_cols - 1)
        return GridPosition(i, j)

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
        return self.get_cell_type(rock_state.position) is RSCellType.GOAL

    def is_valid(self, state):
        if isinstance(state, RockState):
            return self.is_valid_state(state)
        elif isinstance(state, GridPosition):
            return self.is_valid_pos(state)
        else:
            return False

    def is_valid_state(self, rock_state):
        pos = rock_state.position
        return 0 <= pos.i < self.n_rows and 0 <= pos.j < self.n_cols and \
            self.get_cell_type(pos) is not RSCellType.OBSTACLE

    def is_valid_pos(self, pos):
        return 0 <= pos.i < self.n_rows and 0 <= pos.j < self.n_cols and \
            self.get_cell_type(pos) is not RSCellType.OBSTACLE

    ''' --------------- get next position and state ---------------'''
    def make_adjacent_position(self, pos, action_type):
        if action_type is ActionType.NORTH:
            pos.i -= 1
        elif action_type is ActionType.EAST:
            pos.j += 1
        elif action_type is ActionType.SOUTH:
            pos.i += 1
        elif action_type is ActionType.WEST:
            pos.j -= 1
        return pos

    def make_next_position(self, pos, action_type):
        is_legal = True

        if action_type >= ActionType.CHECK:
            pass

        elif action_type is ActionType.SAMPLE:
            # if you took an illegal action and are in an invalid position
            # sampling is not a legal action to take
            if not self.is_valid_pos(pos):
                is_legal = False
            else:
                rock_no = self.get_cell_type(pos)
                if 0 > rock_no or rock_no >= self.n_rocks:
                    is_legal = False
        else:
            old_position = pos.copy()
            pos = self.make_adjacent_position(pos, action_type)
            if not self.is_valid_pos(pos):
                pos = old_position
                is_legal = False
        return pos, is_legal

    ''' --------------- Black Box Dynamics stuff --------------'''
    def make_next_state(self, state, action):
        action_type = action.bin_number
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

        if action_type is ActionType.SAMPLE:
            self.num_times_sampled += 1.0

            rock_no = self.get_cell_type(next_position)
            next_state_rock_states[rock_no] = False

        return RockState(next_position, next_state_rock_states), True

    def make_observation(self, action, next_state):
        # generate new observation if not checking or sampling a rock
        if action.bin_number < ActionType.SAMPLE:
            # Defaults to empty cell and Bad Rock
            obs = RockObservation()
            # self.logger.info("Created Rock Observation - is_good: %s", str(obs.is_good))
            return obs
        elif action.bin_number == ActionType.SAMPLE:
            # The cell is not empty since it contains a rock, and the rock is now "Bad"
            obs = RockObservation(False, False)
            return obs

        # Already sampled this Rock so it is NO GOOD
        if action.rock_no in self.unique_rocks_sampled:
            return RockObservation(False, False)

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

        return RockObservation(observation, False)

    def make_reward(self, state, action, next_state, is_legal):
        if not is_legal:
            return -self.illegal_move_penalty
        if self.is_terminal(next_state):

            # if the agent never sampled any rocks and yet the agent believes there are still good rocks out there,
            # penalize the agent for trying to escape to the exit !!!
            #if not self.sampled_rock_yet and self.any_good_rocks:
            #    return -self.finishing_empty_handed
            #else:
            return self.exit_reward

        if action.bin_number is ActionType.SAMPLE:
            pos = state.position.copy()
            rock_no = self.get_cell_type(pos)
            if 0 <= rock_no < self.n_rocks:
                # If the rock ACTUALLY is good, AND I currently believe it to be good, I get rewarded
                if self.actual_rock_states[rock_no] and state.rock_states[rock_no]:
                    # IMPORTANT - After sampling, the rock is marked as
                    # bad to show that it is has been dealt with
                    # "next states".rock_states[rock_no] is set to False in make_next_state
                    state.rock_states[rock_no] = False
                    self.good_samples += 1.0
                    return self.good_rock_reward
                # otherwise, I either sampled a bad rock I thought was good, sampled a good rock I thought was bad,
                # or sampled a bad rock I thought was bad. All bad behavior!!!
                else:
                    # self.logger.info("Bad rock penalty - %s", str(-self.bad_rock_penalty))
                    self.num_bad_rocks_sampled += 1.0
                    return -self.bad_rock_penalty
            else:
                # self.logger.warning("Invalid sample action on non-existent rock while making reward!")
                return -self.illegal_move_penalty

        return 0

    def generate_reward(self, state, action):
        next_state, is_legal = self.make_next_state(state, action)
        return self.make_reward(state, action, next_state, is_legal)

    def generate_step(self, state, action):
        if action is None:
            print "Tried to generate a step with a null action"
            return None

        result = StepResult()
        result.next_state, is_legal = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state)
        result.reward = self.make_reward(state, action, result.next_state, is_legal)
        result.is_terminal = self.is_terminal(result.next_state)

        return result, is_legal

    ''' ------------ particle generation -----------------'''
    def generate_particles_uninformed(self, previous_belief, action, obs, n_particles):
        old_pos = previous_belief.get_states()[0].position

        particles = []
        while particles.__len__() < n_particles:
            old_state = RockState(old_pos, self.sample_rocks())
            result, is_legal = self.generate_step(old_state, action)
            if obs == result.observation:
                particles.append(result.next_state)
        return particles

    ''' --------------- pretty printing methods --------------- '''
    def disp_cell(self, rs_cell_type):
        if rs_cell_type >= RSCellType.ROCK:
            print hex(rs_cell_type - RSCellType.ROCK),
            return

        if rs_cell_type is RSCellType.EMPTY:
            print ' . ',
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
    def reset(self):
        self.good_samples = 0.0
        self.num_reused_nodes = 0
        self.num_bad_rocks_sampled = 0
        self.num_bad_checks = 0
        self.num_good_checks = 0

    def update(self, step_result):
        if step_result.action.bin_number == ActionType.SAMPLE:
            rock_no = self.get_cell_type(step_result.next_state.position)
            self.unique_rocks_sampled.append(rock_no)
            self.num_times_sampled = 0.0
            self.sampled_rock_yet = True

    def get_legal_actions(self):
        pass

    def get_all_actions_in_order(self):
        all_actions = []
        for code in range(0, 5 + self.n_rocks):
            all_actions.append(RockAction(code))
        return all_actions

    def get_random_action(self):
        """
        Creates a new random RockAction
        :return:
        """
        return RockAction(np.random.random_integers(0, 4 + self.n_rocks))

    def create_action_pool(self):
        return RockActionPool(self)

    def create_root_historical_data(self, solver):
        self.create_new_rock_data()
        return PositionAndRockData(self, self.start_position.copy(), self.all_rock_data, solver)

    def create_new_rock_data(self):
        self.all_rock_data = []
        for i in range(0, self.n_rocks):
            self.all_rock_data.append(RockData())

    def create_observation_pool(self, solver):
        return super(RockModel, self).create_observation_pool(solver)





