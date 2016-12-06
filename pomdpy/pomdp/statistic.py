from __future__ import print_function
from __future__ import division
from builtins import object
from past.utils import old_div
import numpy as np


class Statistic(object):
    """
    General statistics class
    """
    def __init__(self, name, val=0.0, count=0.0):
        self.name = name
        self.mean = val
        self.count = count
        self.variance = 0.0
        self.max = -np.inf
        self.min = np.inf
        self.running_total = 0.0

    def add(self, val):
        self.running_total += val
        mean_old = self.mean
        count_old = self.count

        self.count += 1
        assert self.count > 0
        self.mean += old_div((val - self.mean), self.count)
        self.variance = old_div((count_old * (self.variance + mean_old * mean_old) + val * val), self.count) - self.mean * self.mean
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val

    def std_dev(self):
        return np.sqrt(self.variance)

    def std_err(self):
        return np.sqrt(old_div(self.variance, self.count))

    def clear(self):
        self.mean = 0.0
        self.count = 0
        self.variance = 0.0
        self.max = -np.inf
        self.min = np.inf

    def show(self):
        print("Name = ", self.name)
        print("Running Total = ", self.running_total)
        print("Mean = ", self.mean)
        print("Count = ", self.count)
        print("Variance = ", self.variance)
        print("Max = ", self.max)
        print("Min = ", self.min)