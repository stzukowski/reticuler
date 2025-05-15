"""Preparing plots.

Functions:
    plot_tree(ax, network, height=2.0, width=2.0, \*\*kwargs_tree_plot)

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import MultiLineString, Polygon

from reticuler.utilities.misc import rotation_matrix
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


def plot_polygon(ax, poly, **kwargs):
    """Plot shapely Polygon (or MultiPolygon)"""
    collections = []
    polygons = [poly] if poly.geom_type=="Polygon" else poly.geoms
    for i, poly in enumerate(polygons):
        path = Path.make_compound_path(
            Path(np.asarray(poly.exterior.coords)[:, :2]),
            *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])
    
        patch = PathPatch(path) # , **kwargs)
        collections.append(PatchCollection([patch], **kwargs))
        
        ax.add_collection(collections[-1], autolim=True)
        ax.autoscale_view()
    
    return collections


def plot_tree(ax, system, 
              rot_angle=None, is_thick=None, 
              xmin=None, xmax=None, 
              ymin=None, ymax=None,
              facecolor="#def1ff",
              **kwargs_tree_plot):
    """Plot a tree with optional arguments from \*\*kwargs_tree, and a black box (with facecolor).

    Parameters
    -------
    ax : Axes
        An object to plot on.
    network : object of class Network
        Network to plot.
    xmin : float, default None
        Plot extends from x=``xmin`` to x=``xmax``.
        If None xmin = min( x of the box ).
    xmax : float, default None
        Plot extends from x=``xmin`` to x=``xmax``.
        If None xmax = max( x of the box ).
    ymin : float, default None
        Plot extends from y=``ymin`` to x=``ymax``.
        If None ymin = min( y of the box ).
    ymax : float, default None
        Plot extends from y=``ymin`` to x=``ymax``.
        If None ymin = max( y of the network).
    facecolor : str or tuple, default `#def1ff`
        facecolor for box. To manipulate transparency provide tuple 
        (r,g,b,alpha) (e.g. (0, 0, 0, 0.5) ).
    kwargs_plots : dict, default {`color`: `#0066CC`, `linewidth`: 2.5}
        Arguments to plot the tree.

    Returns
    -------
    None.

    """
    rot_angle = 0 if rot_angle is None else rot_angle
    is_thick = type(system.extender.pde_solver).__name__=="FreeFEM_ThickFingers" \
                        if is_thick is None else is_thick
    if is_thick:
        options_tree_plot = {"color": "0.5", "linewidth": 1.25}
    else:
        options_tree_plot = {"color": "#0066CC", "linewidth": 1.25}
    options_tree_plot.update(kwargs_tree_plot) 
    

    base = ax.transData
    rot = transforms.Affine2D().rotate_deg(-rot_angle)

    # PLOT BOX
    box = Polygon(system.network.box.points)
    plot_polygon(ax, box, 
                 edgecolor="0", 
                 linewidth=options_tree_plot["linewidth"]*2, 
                 facecolor="#f2f2f2ff" if is_thick else facecolor,
                 transform=rot+base)

    # PLOT LINES
    ymax_tree = 2
    pts = [] # list with regularized points (skeleton)
    for branch in system.network.branches:
        line = branch.points
        if np.max(line[:, 1])*1.05 > ymax_tree:
            ymax_tree = 1.05*np.max(line[:, 1])
        # ax.plot(*line.T, **options_tree_plot)
        if not is_thick:
            ax.plot(*line.T, **options_tree_plot, transform=rot+base)
        else:   
            pts.append(branch.points)
    if is_thick:
        # thicken tree and find intersection with the box
        tree = MultiLineString(pts)
        thick_tree = tree.buffer(distance=system.extender.pde_solver.finger_width/2, 
                                 cap_style=1, join_style=1, resolution=99)
        thick_tree = box.intersection(thick_tree)
    
        plot_polygon(ax, thick_tree, transform=rot+base, 
                     edgecolor="0", facecolor=options_tree_plot["color"])

    # if xmin is not None or xmax is not None \
    #     or ymin is not None or ymax is not None:
    xmin = np.min(system.network.box.points[:,0]) if xmin is None else xmin
    xmax = np.max(system.network.box.points[:,0]) if xmax is None else xmax
    ymin = np.min(system.network.box.points[:,1]) if ymin is None else ymin
    ymax_tree = np.min( (np.max(system.network.box.points[:,1]), ymax_tree) )
    ymax = ymax_tree if ymax is None else ymax
    
    xmin, ymin = np.dot(rotation_matrix(rot_angle/180*np.pi), [xmin, ymin])
    xmax, ymax = np.dot(rotation_matrix(rot_angle/180*np.pi), [xmax, ymax])
    ax.set_xlim(min(xmin, xmax), max(xmin, xmax))
    ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
    ax.axis("off")
    ax.set_aspect("equal")


def animate_tree(system0, 
                 rot_angle=None, is_thick=None, 
                 xmin=None, xmax=None, 
                 ymin=None, ymax=None,
                 max_time=None, speed_factor=None, 
                 **kwargs_tree_plot):
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
    rot_angle = 0 if rot_angle is None else rot_angle
    is_thick = type(system0.extender.pde_solver).__name__=="FreeFEM_ThickFingers" \
                        if is_thick is None else is_thick
    if is_thick:
        options_tree_plot = {"color": "0.5", "linewidth": 1.25}
    else:
        options_tree_plot = {"color": "#0066CC", "linewidth": 1.25}
    options_tree_plot.update(kwargs_tree_plot) 
    max_time = system0.growth_gauges[3] if max_time is None else max_time
    speed_factor = 1 if speed_factor is None else speed_factor

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    base = ax.transData
    rot = transforms.Affine2D().rotate_deg(-rot_angle)

    # PLOT BOX
    box = Polygon(system0.network.box.points)
    plot_polygon(ax, box, 
                 edgecolor="0", 
                 linewidth=options_tree_plot["linewidth"]*2, 
                 facecolor="0.8" if is_thick else "#def1ff",
                 transform=rot+base)

    ymax_tree = 2
    artists = []
    for i in np.linspace(0, max_time, int(200/speed_factor)):
        system = system0.copy()
        clippers.clip_to_time(system, i)
        
        etwas = [] # for tree buffer
        if type(system.morpher).__name__ == "Leaf":
            # update box
            box = Polygon(system.network.box.points)
            etwas.append(*plot_polygon(ax, box, 
                         edgecolor="0", 
                         linewidth=options_tree_plot["linewidth"]*2, 
                         facecolor="0.8" if is_thick else "#def1ff",
                         transform=rot+base))
        
        
        # PLOT LINES
        ymax_tree = 2
        for branch in system.network.branches:
            line = branch.points
            if np.max(line[:, 1])*1.05 > ymax_tree:
                ymax_tree = 1.05*np.max(line[:, 1])
            # ax.plot(*line.T, **options_tree_plot)
            if not is_thick:
                l = ax.plot(*line.T, **options_tree_plot, transform=rot+base)
                etwas.append(l[0])
            else:   
                etwas.append(branch.points)
        if is_thick:
            # thicken tree and find intersection with the box
            tree = MultiLineString(etwas)
            thick_tree = tree.buffer(distance=system.extender.pde_solver.finger_width/2, 
                                     cap_style=1, join_style=1, resolution=99)
            thick_tree = box.intersection(thick_tree)
        
            # fig, ax = plt.subplots()
            etwas = plot_polygon(ax, thick_tree, transform=rot+base, 
                         edgecolor="0", facecolor=options_tree_plot["color"])
            
        artists.append(etwas)
        
    if xmin is not None or xmax is not None \
        or ymin is not None or ymax is not None:
        xmin = np.min(system.network.box.points[:,0]) if xmin is None else xmin
        xmax = np.max(system.network.box.points[:,0]) if xmax is None else xmax
        ymin = np.min(system.network.box.points[:,1]) if ymin is None else ymin
        ymax_tree = np.min( (np.max(system.network.box.points[:,1]), ymax_tree) )
        ymax = ymax_tree if ymax is None else ymax
        
        xmin, ymin = np.dot(rotation_matrix(rot_angle/180*np.pi), [xmin, ymin])
        xmax, ymax = np.dot(rotation_matrix(rot_angle/180*np.pi), [xmax, ymax])
        ax.set_xlim(min(xmin, xmax), max(xmin, xmax))
        ax.set_ylim(min(ymin, ymax), max(ymin, xmax))
        fig.set_size_inches(abs(xmax-xmin), abs(ymax-ymin))
    ax.axis("off")
    ax.set_aspect("equal")

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=20, blit=True)
    return ani