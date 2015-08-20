__author__ = 'patrickemami'

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

def do_bar_plot(name, left, height, **kwargs):
    plt.bar(left, height, kwargs)
    plt.savefig(name)

def do_scatter(name, left, height, **kwargs):
    plt.scatter(left, height, kwargs)
    plt.savefig(name)