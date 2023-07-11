"""Preparing plots.

Functions:
    plot_tree(ax, network, height=2.0, width=2.0, \*\*kwargs_tree_plot)

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
from reticuler.extending_kernels.extenders import rotation_matrix
from reticuler.user_interface import clippers

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

    # PLOT BOX
    points_to_plot = network.box.points[network.box.connections]
    for pts in points_to_plot:
        ax.plot(*pts.T, linewidth=options_tree_plot["linewidth"] * 2, color="0")
    x_max = np.max(points_to_plot[...,0])
    # PLOT LINES
    y_max = 2
    for branch in network.branches:
        line = branch.points
        if np.max(line[:, 1]) > y_max:
            y_max = 1.05*np.max(line[:, 1])
        ax.plot(*line.T, **options_tree_plot)

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

def animate_tree(system0, ylim=None, xlim=None, rot_angle=None, max_step=None, speed_factor=None, **kwargs_tree_plot):
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
    rot_angle = 90 if rot_angle is None else rot_angle
    max_step = system0.growth_gauges[0] if max_step is None else max_step
    speed_factor = 1 if speed_factor is None else speed_factor

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    base = ax.transData
    rot = transforms.Affine2D().rotate_deg(-rot_angle)

    # PLOT BOX
    points_to_plot = system0.network.box.points[system0.network.box.connections]
    for pts in points_to_plot:
        ax.plot(*pts.T, linewidth=options_tree_plot["linewidth"] * 2, color="0", transform=rot+base)
    x_max = np.max(points_to_plot[...,0])

    y_max = 2
    artists = []
    for i in np.linspace(0, max_step, int(200/speed_factor), dtype=int):
        system = system0.copy()
        clippers.clip_to_step(system, i)
        ll = []
        # PLOT LINES
        for branch in system.network.branches:
            line = branch.points
            if np.max(line[:, 1]) > y_max:
                y_max = 1.05*np.max(line[:, 1])
            if type(system.extender.pde_solver).__name__ == "FreeFEM":
                l = ax.plot(*line.T, **options_tree_plot, transform=rot+base)
            else:   
                contours  = system0.extender.pde_solver.finger_contour(branch, system.network)
                contour = np.vstack(contours)
                l2 = ax.fill(contour[:,0], contour[:,1], 
                             color=options_tree_plot["color"], 
                             lw=0, transform=rot+base)
                ll.append(l2[0])
                l = ax.plot(*contour.T, **options_tree_plot, transform=rot+base) # ,marker=".",ms=5
            ll.append(l[0])
        artists.append(ll)
    y_max = np.min( (np.max(points_to_plot[...,1]), y_max) )
        
    xlim = x_max if xlim is None else xlim
    ylim = y_max if ylim is None else ylim
    xlim, ylim = np.dot(rotation_matrix(rot_angle/180*np.pi), [xlim, ylim])
    fig.set_size_inches(abs(xlim), abs(ylim))

    ax.set_xlim(min(0, xlim), max(0, xlim))
    ax.set_ylim(min(0, ylim), max(0, ylim))
    ax.axis("off")
    ax.set_aspect("equal")
    # colouring background
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1)
    ax.set_facecolor("#def1ff")

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=20, blit=True)
    return ani