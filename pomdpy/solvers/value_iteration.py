from solver import Solver
from scipy.optimize import linprog
import numpy as np


class AlphaVector(object):
    def __init__(self, a, v):
        self.action = a
        self.v = v


class ValueIteration(Solver):

    def __init__(self, agent):
        super(ValueIteration, self).__init__(agent)
        self.gamma = set()

    def simulate(self, belief_state, eps, start_time):
        pass

    @staticmethod
    def reset(agent, model):
        return ValueIteration(agent)

    def select_action(self, eps, start_time):
        pass

