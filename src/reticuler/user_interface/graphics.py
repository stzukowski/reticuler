"""Preparing plots.

Functions:
    plot_tree(ax, network, height=2.0, width=2.0, \*\*kwargs_tree_plot)

"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "text.usetex": True,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.titlesize": 12,
        "axes.labelsize": 8,
        "lines.linewidth": 1,
        "lines.markersize": 1,
        "lines.marker": "None",
        "lines.solid_capstyle": "round",
        "font.family": "serif",
        "font.serif": ["Computer Modern"],  # ['Times New Roman']
    }
)
# 'font.size': 20,
# 'font.family': 'serif',
# 'font.serif': ['Times New Roman']})
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
cm2inch = 1 / 2.54
golden = (1 + 5**0.5) / 2


def plot_tree(ax, network, ylim=None, xlim=None, **kwargs_tree_plot):
    """Plot a tree with optional arguments from \*\*kwargs_tree, and a black box.

    Parameters
    -------
    ax : Axes
        An object to plot on.
    network : object of class Network
        Network to plot.
    ylim : float, default None
        Plot extends from y=0 to y=``ylim``.
        If None ylim = max( 2, max height of the network ).
    xlim : float, default 2.0
        Plot extends from x=0 to x=``xlim``.
    kwargs_plots : dict, default {`color`: `#0066CC`, `linewidth`: 2.5}
        Arguments to plot the tree.

    Returns
    -------
    None.

    """
    options_tree_plot = {"color": "#0066CC", "linewidth": 1.25}
    options_tree_plot.update(kwargs_tree_plot)

    # PLOT LINES
    y_max = 2
    for branch in network.branches:
        line = branch.points
        if np.max(line[:, 1]) > y_max:
            y_max = 1.05*np.max(line[:, 1])
        ax.plot(*line.T, **options_tree_plot, marker='.', ms=5)
    # PLOT BOX
    points_to_plot = network.box.points[network.box.connections]
    for pts in points_to_plot:
        ax.plot(*pts.T, linewidth=options_tree_plot["linewidth"] * 2, color="0")
    x_max = np.max(points_to_plot[...,0])

    ax.axis("off")
    # colouring background
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1)
    ax.set_facecolor("#def1ff")
    ax.set_aspect("equal")
    xlim = x_max if xlim is None else xlim
    ax.set_xlim(0, xlim)
    ylim = y_max if ylim is None else ylim
    ax.set_ylim(0, ylim)    
