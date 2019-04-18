from __future__ import absolute_import
from .solver import Solver
from .alpha_vector import AlphaVector
from scipy.optimize import linprog
import numpy as np
from itertools import product


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
            print('[Value Iteration] planning horizon {}...'.format(k))
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
                                v_new[idx][u][z][i] += v.v[i] * o[u][i][z] * t[u][j][i]
                idx += 1
            # add (|A| * |V|^|Z|) alpha-vectors to gamma, |V| is |gamma_k|
            for u in range(actions):
                c = self.compute_indices(idx, observations)
                for indices in c:  # n elements in c is |V|^|Z|
                    for z in range(observations):
                        temp = np.zeros(states)
                        for i in range(states):
                            temp[i] = discount * (r[u][i] + v_new[indices[z]][u][z][i])
                        gamma_k.add(AlphaVector(a=u, v=temp))
            self.gamma.update(gamma_k)
            if first:
                # remove the dummy alpha vector
                self.gamma.remove(dummy)
                first = False
            self.prune(states)
            #  plot_gamma(title='V(b) for horizon T = ' + str(k + 1), self.gamma)

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
            av_i = F.pop()  # get a reference to av_i
            F.add(av_i)  # don't want to remove it yet from F
            dominated = False
            for av_j in Q:
                c = np.append(np.zeros(n_states), [1.])
                A_ub = np.array([np.append(-(av_i.v - av_j.v), [-1.])])
                b_ub = np.array([-delta])

                res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
                if res.x[n_states] > 0.0:
                    # this one is dominated
                    dominated = True
                    F.remove(av_i)
                    break

            if not dominated:
                max_k = -np.inf
                best = None
                for av_k in F:
                    b = res.x[0:2]
                    v = np.dot(av_k.v, b)
                    if v > max_k:
                        max_k = v
                        best = av_k
                F.remove(best)
                if not self.check_duplicate(Q, best):
                    Q.update({best})
        self.gamma = Q

    @staticmethod
    def check_duplicate(a, av):
        """
        Check whether alpha vector av is already in set a

        :param a:
        :param av:
        :return:
        """
        for av_i in a:
            if np.allclose(av_i.v, av.v):
                return True
            if av_i.v[0] == av.v[0] and av_i.v[1] > av.v[1]:
                return True
            if av_i.v[1] == av.v[1] and av_i.v[0] > av.v[0]:
                return True

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
