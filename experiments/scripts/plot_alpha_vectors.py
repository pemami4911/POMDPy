import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches
import numpy as np


def plot_alpha_vectors(title, gamma, n_actions):
    """
    Plot the current set of alpha vectors over the belief simplex
    :return:
    """
    fig = plt.figure()
    plt.title(title)
    pts = 30
    x = np.linspace(0., 1., num=pts)
    # y = np.linspace(0., 1., num=pts)
    # Z = np.zeros(shape=(pts, pts))
    # X, Y = np.meshgrid(x, y)
    y = np.zeros(shape=pts)
    cmap = get_cmap(n_actions * 10)
    patches, patches_handles = [], []
    for i in range(n_actions):
        patches.append(cmap(i * 10))
        patches_handles.append(mpatches.Patch(color=patches[i], label='action {}'.format(i)))

    for av in gamma:
        for i in range(pts):
            # for j in range(pts):
            y[i] = np.dot(av.v, np.array([x[i], 1. - x[i]]))
        plt.plot(x, y, color=patches[av.action], linewidth=2)

    plt.xlabel('p1')
    plt.legend(handles=patches_handles)

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
