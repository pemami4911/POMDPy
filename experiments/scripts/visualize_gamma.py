from __future__ import absolute_import
from experiments.scripts import plot_alpha_vectors
from experiments.scripts import pickle_wrapper
import os

if __name__ == '__main__':
    n_actions = 3
    my_dir = os.path.dirname(__file__)
    weight_dir = os.path.join(my_dir, '..', '..', 'experiments', 'pickle_jar')

    gamma = pickle_wrapper.load_pkl(os.path.join(weight_dir, 'linear_alpha_net_vectors.pkl'))

    plot_alpha_vectors.plot_alpha_vectors('Alpha vectors computed with linear function approximation', gamma, n_actions)
