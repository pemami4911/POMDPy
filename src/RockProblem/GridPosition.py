__author__ = 'patrickemami'

import numpy as np

class GridPosition():

    '''
    Supported Constructors - GripPosition(i,j) or GridPosition() - defaults to 0,0
    '''
    def __init__(self, i=0, j=0):
        if i is None and j is None:
            self.i = 0
            self.j = 0
        else:
            self.i = i
            self.j = j

    def print_position(self):
        print '(',
        print self.i,
        print ',',
        print self.j,
        print ')'

    def copy(self):
        return GridPosition(self.i, self.j)

    def equals(self, other_position):
        assert isinstance(other_position, GridPosition)
        return self.i == other_position.i and self.j == other_position.j

    def as_list(self):
        return [self.i, self.j]

    def manhattan_distance(self, other_position):
        assert isinstance(other_position, GridPosition)
        return np.linalg.norm(np.subtract(self.as_list(), other_position.as_list()), 1)

    def euclidean_distance(self, other_position):
        assert isinstance(other_position, GridPosition)
        return np.sqrt(np.power(other_position.j - self.j, 2.0) + np.power(other_position.i - self.i, 2.0))
