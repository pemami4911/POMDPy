from __future__ import absolute_import
from builtins import range
from builtins import object
from .solver import Solver
from scipy.optimize import linprog
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D


class AlphaVector(object):
    def __init__(self, a, v):
        self.action = a
        self.v = v

    def copy(self):
        return AlphaVector(self.action, self.v)


class ValueIteration(Solver):
    def __init__(self, agent):
        """
        Initialize the POMDP exact value iteration solver
        :param agent:
        :return:
        """
        super(ValueIteration, self).__init__(agent)
        self.gamma = set()
        self.history = agent.histories.create_sequence()

    @staticmethod
    def reset(agent):
        return ValueIteration(agent)

    def value_iteration(self, t, o, r, horizon):
        """
        Solve the POMDP by computing all alpha vectors
        :param t: transition probability matrix
        :param o: observation probability matrix
        :param r: immediate rewards matrix
        :param horizon: integer valued scalar represented the number of planning steps
        :return:
        """
        discount = self.model.discount
        actions = len(self.model.get_all_actions())  # |A| actions
        states = self.model.num_states  # |S| states
        observations = len(self.model.get_all_observations())  # |Z| observations
        first = True

        # initialize gamma with a 0 alpha-vector
        dummy = AlphaVector(a=-1, v=np.zeros(states))
        self.gamma.add(dummy)

        # start with 1 step planning horizon, up to horizon-length planning horizon
        for k in range(horizon):
            # new set of alpha vectors to add to set gamma
            gamma_k = set()
            # Compute the new coefficients for the new alpha-vectors
            v_new = np.zeros(shape=(len(self.gamma), actions, observations, states))
            idx = 0
            for v in self.gamma:
                for u in range(actions):
                    for z in range(observations):
                        for j in range(states):
                            for i in range(states):
                                # v_i_k * p(z | x_i, u) * p(x_i | u, x_j)
                                v_new[idx][u][z][i] = v.v[i] * o[u][i][z] * t[u][j][i]
                idx += 1
            # add (|A| * |V|^|Z|) alpha-vectors to gamma, |V| is |gamma_k|
            for u in range(actions):
                c = self.compute_indices(idx, observations)
                for indices in c:
                    temp = np.zeros(states)
                    for i in range(states):
                        for z in range(observations):
                            temp[i] = discount * (r[u][i] + v_new[indices[z]][u][z][i])
                    gamma_k.add(AlphaVector(a=u, v=temp))
            self.gamma.update(gamma_k)
            if first:
                # remove the dummy alpha vector
                self.gamma.remove(dummy)
                first = False
            self.prune(states)
            #  self.plot_gamma(title='V(b) for horizon T = ' + str(k + 1))

    @staticmethod
    def compute_indices(k, m):
        """
        Compute all orderings of m elements with values between [0, k-1]
        :param k: Number of alpha-vectors
        :param m: Number of observations
        :return: list of lists, where each list contains m elements, and each element is in [0, k-1].
        Total should be k^m elements
        """
        x = list(range(k))
        return [p for p in product(x, repeat=m)]

    def prune(self, n_states):
        """
        Remove dominated alpha-vectors using Lark's filtering algorithm
        :param n_states
        :return:
        """
        # parameters for linear program
        delta = 0.0000000001
        # equality constraints on the belief states
        A_eq = np.array([np.append(np.ones(n_states), [0.])])
        b_eq = np.array([1.])

        # dirty set
        F = self.gamma.copy()
        # clean set
        Q = set()

        for i in range(n_states):
            max_i = -np.inf
            best = None
            for av in F:
                if av.v[i] > max_i:
                    max_i = av.v[i]
                    best = av
            Q.update({best})
            F.remove(best)
        while F:
            av_i = F.pop()
            dominated = False
            for av_j in Q:
                c = np.append(np.zeros(n_states), [1.])
                A_ub = np.array([np.append(-(av_i.v - av_j.v), [-1.])])
                b_ub = np.array([-delta])

                res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
                if res.x[n_states] > 0.0:
                    # this one is dominated
                    dominated = True
                    break

            if not dominated:
                Q.update({av_i})

        self.gamma = Q

    def plot_gamma(self, title):
        """
        Plot the current set of alpha vectors over the belief simplex
        :return:
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.title(title)
        pts = 20
        x = np.linspace(0., 1., num=pts)
        y = np.linspace(0., 1., num=pts)
        Z = np.zeros(shape=(pts, pts))
        X, Y = np.meshgrid(x, y)
        cmap = self.get_cmap(len(self.gamma))
        color_idx = 0
        for av in self.gamma:
            for i in range(pts):
                for j in range(pts):
                    Z[i][j] = np.dot(av.v, np.array([x[i], y[j]]))

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color=cmap(color_idx), linewidth=0, antialiased=False)
            color_idx += 1
        plt.xlabel('p1')
        plt.ylabel('p2')
        plt.show()

    @staticmethod
    def get_cmap(N):
        """
        Returns a function that maps each index in 0, 1, ... N-1 to a distinct
        RGB color.
        """
        color_norm = colors.Normalize(vmin=0, vmax=N - 1)
        scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

        def map_index_to_rgb_color(index):
            return scalar_map.to_rgba(index)

        return map_index_to_rgb_color

    @staticmethod
    def select_action(belief, vector_set):
        """
        Compute optimal action given a belief distribution
        :param belief: dim(belief) == dim(AlphaVector)
        :param vector_set
        :return:
        """
        max_v = -np.inf
        best = None
        for av in vector_set:
            v = np.dot(av.v, belief)
            if v > max_v:
                max_v = v
                best = av
        if best is None:
            raise ValueError('Vector set should not be empty')

        return best.action, best
