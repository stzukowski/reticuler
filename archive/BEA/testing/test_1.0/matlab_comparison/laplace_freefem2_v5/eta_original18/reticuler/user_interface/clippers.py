"""Clipping a network.

Functions:
    clip_to_step(system, max_step)
    clip_to_length(system, max_length)
    clip_to_height(system, max_height)

"""
import numpy as np

from reticuler.system import System
from reticuler.backward_evolution.system_back import BackwardSystem, BackwardBranch

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
    
def clip_to_BEA_step(system, backward_system, max_BEA_step):
    branches_to_iterate = backward_system.system.network.branches.copy()
    # active_inds = np.empty(len(branches_to_iterate), dtype=int)
    # sleeping_inds = np.empty(len(branches_to_iterate), dtype=int)
    max_BEA_step_found = 0
    for i, forward_branch in enumerate(branches_to_iterate):
        backward_branch = backward_system.backward_branches[forward_branch.ID]
        # active_inds[i] = backward_branch.active_ind
        # sleeping_inds[i] = backward_branch.sleeping_ind
        if len(backward_branch.steps):
            ind_backward = np.sum(backward_branch.steps<=max_BEA_step)-1
            if ind_backward<0:
                # branch hasn't been trimmed at all -- delete BEA results
                backward_system.backward_branches[forward_branch.ID] =\
                    BackwardBranch(backward_branch.ID, backward_branch.mother_ID)
                # and update backward_bifurcations
                ind_bifurcation = backward_system.backward_bifurcations.mother_IDs==backward_branch.mother_ID
                if backward_system.backward_bifurcations.flags[ind_bifurcation] == 3:
                    backward_system.backward_bifurcations.flags[ind_bifurcation] = 1
                    backward_system.backward_bifurcations.a1a2a3_coefficients[ind_bifurcation,...] = 0
                elif backward_system.backward_bifurcations.flags[ind_bifurcation] < 3:
                    backward_system.backward_bifurcations.flags[ind_bifurcation] = 0
                    backward_system.backward_bifurcations.length_mismatch[ind_bifurcation] = 0
            else:
                max_BEA_step_found = max(max_BEA_step_found, \
                                         min(max_BEA_step, backward_branch.steps[ind_backward]))
                # max_BEA_step_found = 1 
                num_left = backward_branch.nums_left[ind_backward]
                if num_left==0:
                    # whole branch trimmed - delete forward_branch
                    mask = backward_system.system.network.branch_connectivity[:,1]!=forward_branch.ID
                    backward_system.system.network.branch_connectivity = backward_system.system.network.branch_connectivity[mask,...]
                    backward_system.system.network.branches.remove(forward_branch)
                else:
                    # branch partially trimmed - trim accordingly
                    forward_branch.steps = forward_branch.steps[:num_left+1]
                    forward_branch.points = forward_branch.points[:num_left+1]
                    forward_branch.points[-1] = backward_branch.points[ind_backward]
                    backward_branch.points = backward_branch.points[:ind_backward+1]
                    backward_branch.nums_left = backward_branch.nums_left[:ind_backward+1]
                    backward_branch.steps = backward_branch.steps[:ind_backward+1]
                    backward_branch.a1a2a3_coefficients = backward_branch.a1a2a3_coefficients[:ind_backward+1]
                    backward_branch.overshoot = backward_branch.overshoot[:ind_backward+1]
                    backward_branch.angular_deflection = backward_branch.angular_deflection[:ind_backward+1]
                    # and update backward_bifurcations
                    ind_bifurcation = backward_system.backward_bifurcations.mother_IDs==backward_branch.mother_ID
                    if backward_system.backward_bifurcations.flags[ind_bifurcation] == 3:
                        backward_system.backward_bifurcations.flags[ind_bifurcation] = 1
                        backward_system.backward_bifurcations.a1a2a3_coefficients[ind_bifurcation,...] = 0
                    elif backward_system.backward_bifurcations.flags[ind_bifurcation] == 1:
                        backward_system.backward_bifurcations.flags[ind_bifurcation] = 0
                        backward_system.backward_bifurcations.length_mismatch[ind_bifurcation] = 0
                    # print(backward_branch.active_ind)
    backward_system.import_branches(backward_system.system.network)
    backward_system.BEA_step = max_BEA_step_found
    # backward_system.system.network.active_branches = [branches_to_iterate[i] for i in np.argsort(active_inds)[np.sort(active_inds)>-1]]
    # backward_system.system.network.sleeping_branches = [branches_to_iterate[i] for i in np.argsort(sleeping_inds)[np.sort(sleeping_inds)>-1]]
    

if __name__ == "__main__":
    # Import System from JSON file
    dir_name = "G:/My Drive/Research/Network simulations/reticuler/archive/BEA/testing/test_1.0/matlab_comparison/laplace_freefem2_v5/eta_original18/debug_eta0/"
    system = System.import_json(input_file=dir_name+"original_tree_0")
    backward_system = BackwardSystem.import_json(input_file=dir_name+"eta00_back_0", system=system)
    backward_system.exp_name = dir_name+"eta00_back"
    backward_system.system.exp_name = dir_name+"original_tree"

    clip_to_BEA_step(system, backward_system, max_BEA_step=23)
    print(backward_system.backward_branches[-1].steps)

    # backward_system.system.export_json()         
    # backward_system.export_json()      
        
    # import matplotlib.pyplot as plt
    # from reticuler.user_interface import graphics
    # fig, ax = plt.subplots()
    # graphics.plot_tree(
    #     ax,
    #     network=system.network,
    #     ylim=2.5,
    #     # color='{}'.format(step/lim)
    # )    
    
    # Plot different stages of the evolution
    # import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # from reticuler.user_interface import graphics
    # fig, ax = plt.subplots()
    # lim = system.growth_gauges[0]+1
    # for step in reversed(np.linspace(100, lim, 30)):
    #     # print(step)
    #     clip_to_step(system, step)
    #     graphics.plot_tree(
    #         ax,
    #         network=system.network,
    #         ylim=10.5,
    #         color=mpl.colormaps['copper'](step/lim),
    #         # color='{}'.format(step/lim)
    #     )
    