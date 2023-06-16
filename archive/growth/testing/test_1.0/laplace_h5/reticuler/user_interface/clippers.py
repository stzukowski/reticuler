"""Clipping a network.

Functions:
    clip_to_step(system, max_step)
    clip_to_length(system, max_length)
    clip_to_height(system, max_height)

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from reticuler.system import System
from reticuler.user_interface import graphics

def clip_to_step(system, max_step):
    branches_to_iterate = system.network.branches.copy()
    for branch in branches_to_iterate:
        to_trash  = branch.steps > max_step
        if sum(to_trash)+1 >= len(branch.steps):
            mask = system.network.branch_connectivity[:,1]!=branch.ID
            if sum(~mask)==1:
                mother_ID = system.network.branch_connectivity[~mask,0][0]
                mother_branch = [b for b in branches_to_iterate if b.ID==mother_ID][0]
                if not mother_branch in system.network.active_branches:
                    system.network.active_branches.append(mother_branch)
            system.network.branch_connectivity = system.network.branch_connectivity[mask,...]
            system.network.branches.remove(branch)
            if branch in system.network.active_branches:
                system.network.active_branches.remove(branch)
            if branch in system.network.sleeping_branches:
                system.network.sleeping_branches.remove(branch)
        else:
            branch.points = branch.points[~to_trash]
            branch.steps = branch.steps[~to_trash]
            
    system.growth_gauges[0] = max_step
    system.growth_gauges[1], system.growth_gauges[2] = system.network.height_and_length()
    system.growth_gauges[3] = 0


def clip_to_length(system, max_length):
    length_throughout_evolution = np.zeros(int(system.growth_gauges[0])+1)
    branches_to_iterate = system.network.branches.copy()
    for branch in branches_to_iterate:
        length_throughout_evolution[branch.steps[1:]] = \
            length_throughout_evolution[branch.steps[1:]] + \
            np.linalg.norm(branch.points[1:]-branch.points[:-1], axis=1)
    length_throughout_evolution = np.cumsum(length_throughout_evolution)
    
    clip_to_step(system, sum(length_throughout_evolution<max_length))

def clip_to_height(system, max_height):
    all_points_steps = np.zeros(3)
    branches_to_iterate = system.network.branches.copy()
    for branch in branches_to_iterate:
        all_points_steps = np.vstack( ( all_points_steps, branch.points_steps() ) )
    all_points_steps = all_points_steps[1:]
    
    clip_to_step(system, np.min(all_points_steps[all_points_steps[:,1]>max_height, 2]))

if __name__ == "__main__":
    # Import System from JSON file
    system = System.import_json(input_file='laplace/eta18')
    
    fig, ax = plt.subplots()
    lim = system.growth_gauges[0]+1
    for step in reversed(np.linspace(100, lim,30)):
        # print(step)
        clip_to_step(system, step)
        graphics.plot_tree(
            ax,
            network=system.network,
            ylim=10.5,
            color=mpl.colormaps['copper'](step/lim),
            # color='{}'.format(step/lim)
        )