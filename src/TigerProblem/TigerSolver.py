__author__ = 'patrickemami'

from Solvers import Solver


class TigerSolver(Solver.Solver):

    def __init__(self, model):
        super(TigerSolver, self).__init__(model, model.config["policy_representation"])

    def generate_episodes(self, n_particles, root_node):
        pass

    def generate_policy(self):
        pass

    def execute(self):
        pass

