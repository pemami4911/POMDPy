__author__ = 'patrickemami'

import abc

class A():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def foo(self):
        pass

class B(A):

    @abc.abstractmethod
    def foo(self):
        """
        :return:
        """

    @abc.abstractmethod
    def bar(self):
        """
        :return:
        """

class C(B):

    def foo(self):
        print 'C: foo'

    def bar(self):
        print 'C: bar'

my_c = C()
my_c.foo()
my_c.bar()

'''
my_b = B()
my_b.foo()
my_b.bar()
'''
from rock_action import *

r = RockAction(2)
r.print_action()

import rock_state
import grid_position

print 'Testing "isinstance"'
g = grid_position(3,3)
states = [1, 0, 0, 1]

r = rock_state(g, states)
r.print_state()

g2 = grid_position(4,4)
assert isinstance(g2, grid_position)
print "Passed"

states2 = [1, 0, 0, 1]

r2 = rock_state(g2, states2)
print r.equals(r2)

print "distance to: should be 2"
assert 2 != r.distance_to(r2)
print "Passed"

#assert type(states2) is list

print 'manhattan distance test'
import numpy

a = [1,1]
b = [3,3]

assert 4 == numpy.linalg.norm(numpy.subtract(b,a), 1)
print "Passed"

print 'manhattan distance test with GridPosition'
import numpy

a = grid_position(0,0)
b = grid_position(1,3)

assert 4 == a.manhattan_distance(b)
print "Passed"

print 'Euclidean distance test with Grid Position'
a = grid_position(0,0)
b = grid_position(2,2)
assert numpy.sqrt(8) == a.euclidean_distance(b)
print a.euclidean_distance(b)
print "Passed"

print 'Testing "if nots"'

assert not 1 == 2
print "Passed"

print 'Testing "numpy.isfinite"'
test_inf = -numpy.inf
assert not numpy.isfinite(test_inf)
print "Passed"


print 'Testing dictionary of DiscreteActionMappingEntry objects'

<<<<<<< HEAD
from discretePOMDP.DiscreteActionMapping import DiscreteActionMappingEntry
=======
from discretePOMDP.discrete_action_mapping import DiscreteActionMappingEntry
>>>>>>> [issue #1] log files in wrong place

a = DiscreteActionMappingEntry()
b = DiscreteActionMappingEntry()

entries = {0:a, 1:b}

assert a == entries.get(0)
print "Passed"

print 'Testing mutating a dictionary'

c = entries.get(0)

c.bin_number = 100

assert 100 == entries.get(0).bin_number
print "Passed"

print "Testing lazy creation of dictionaries"

entries2 = {}

for i in range(0, 2):
    test_entry = DiscreteActionMappingEntry()
    entries2.__setitem__(i, test_entry)

print entries2.values()
print "Passed"

print "Test random shuffle"

import random

test_list = range(0, 5)
old_list = range(0,5)
random.shuffle(test_list)
assert old_list is not test_list
print "Passed"

print "Testing RockModel creation"

import rock_model

r = rock_model.RockModel("rockproblem")

assert isinstance(r, rock_model.RockModel)
print "Passed"

print "Testing 'sample_state_uninformed and print_state"

print "Initial pos1: ",
r.start_position.print_position()

print "Initial pos2: ",
r.sample_state_uninformed().position.print_position()

print "Passed"

print "Testing 'is_terminal'"
state = r.sample_an_init_state()
state.print_state()
state.position.print_position()
assert r.get_cell_type(state.position) is not -2
assert r.is_terminal(state) is not True
print "Passed"

r.draw_env()

print "------------- Black Box Dynamics Testing ----------------"
current_state = r.sample_state_uninformed()
current_action = numpy.random.choice(r.get_all_actions())
print "Current state: ",
current_state.print_state(),
print "Current position: ",
current_state.position.print_position()
print "Current action: ",
current_action.print_action()

next_state, is_legal = r.make_next_state(current_state, current_action)

print "Next state: ",
next_state.print_state()
print "Current position: ",
next_state.position.print_position()
print "Is legal: ",
print is_legal

next_observation = r.make_observation(current_action, next_state)

print "Next observation: ",
next_observation.print_observation()

print "Make next reward: ",

next_reward = r.make_reward(current_state, current_action, next_state, is_legal)

print next_reward

print "Get cell type test: ",
print r.get_cell_type(current_state.position)

print "Passed"

print "Testing Step Generation"

import model

step = r.generate_step(next_state, numpy.random.choice(r.get_all_actions()))
assert isinstance(step[0], model.StepResult)
step[0].print_step_result()

print "------------- End Black Box Dynamics Testing ----------------"

g = grid_position(1, 2)
h = [grid_position(1, 2)]

if g in h:
    print "SUCCESS"
else:
    print "FAILURE"

if g == grid_position(1, 2):
    print "SUCCESS"
else:
    print "FAILURE"
