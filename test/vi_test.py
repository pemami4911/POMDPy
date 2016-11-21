from scipy.optimize import linprog
import numpy as np

if __name__ == '__main__':

    DELTA = 0.0000000001
    A_eq = np.array([[1., 1., 0.]])
    b_eq = np.array([1.])

    av1 = np.array([-100, 100])
    av2 = np.array([-40, -5])
    av3 = np.array([40, 55])
    av4 = np.array([100, -50])

    g = [av1, av2, av3, av4]

    for i in g:
        for j in g:
            if np.array_equal(i, j):
                continue
            c = np.array([0, 0, 1])
            A_ub = np.array([np.append(-(i - j), [-1.])])
            b_ub = np.array([-DELTA])

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))

            if res.x[2] > 0.0:
                assert np.array_equal(i, np.array([-40, -5]))
                assert np.array_equal(j, np.array([40, 55]))
