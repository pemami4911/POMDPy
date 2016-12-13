from __future__ import print_function
from builtins import str
from builtins import object
import numpy as np


class GridPosition(object):
    """
    Supported Constructors - GripPosition(i,j) or GridPosition() - defaults to 0,0
    """
    def __init__(self, i=0, j=0):
        if i is None and j is None:
            self.i = 0
            self.j = 0
        else:
            self.i = i
            self.j = j

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def print_position(self):
        print('(', end=' ')
        print(self.i, end=' ')
        print(',', end=' ')
        print(self.j, end=' ')
        print(')')

    def to_string(self):
        return '(' + str(self.i) + ',' + str(self.j) + ')'

    def copy(self):
        return GridPosition(self.i, self.j)

    def as_list(self):
        return [self.i, self.j]

    def manhattan_distance(self, other_position):
        return np.linalg.norm(np.subtract(self.as_list(), other_position.as_list()), 1)

    def euclidean_distance(self, other_position):
        return np.sqrt(np.power(other_position.j - self.j, 2.0) + np.power(other_position.i - self.i, 2.0))
