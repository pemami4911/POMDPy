import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('PS')


def do_bar_plot(name, left, height, **kwargs):
    plt.bar(left, height, kwargs)
    plt.savefig(name)


def do_scatter(name, left, height, **kwargs):
    plt.scatter(left, height, kwargs)
    plt.savefig(name)
