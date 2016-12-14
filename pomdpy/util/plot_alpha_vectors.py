import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_gamma(title, gamma):
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
    cmap = get_cmap(len(gamma))
    color_idx = 0
    for av in gamma:
        for i in range(pts):
            for j in range(pts):
                Z[i][j] = np.dot(av.v, np.array([x[i], y[j]]))

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color=cmap(color_idx), linewidth=0, antialiased=False)
        color_idx += 1
    plt.xlabel('p1')
    plt.ylabel('p2')
    plt.show()


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
