"""Building blocks of the System.

Classes:
    Box
    Branch
    Network
"""

import logging
import numpy as np
import copy

from reticuler.utilities.geometry.branch import Branch

logger = logging.getLogger("reticuler")
from reticuler.utilities.misc import DIRICHLET_0, DIRICHLET_0_GLOB_FLUX, NEUMANN_0
from reticuler.utilities.misc import find_reconnection_point

class Network:
    """A class of network with its branches and containing box.

    Attributes
    ----------
    box : Box
        An object of class Box.
    branches : list, default []
        A list of all branches (objects of class Branch) composing the network.
    active_branches : list, default []
        A list of all branches that still extend.
    sleeping_branches : list, default []
        A list of all branches without enough
        flux to move (may revive in the Poisson case - TO DO).
    branch_connectivity : array, default []
        A 2-n array with connections between the branches
        (noted with branch IDs).

    """

    def __init__(
        self,
        box,
        branches=None,
        active_branches=None,
        sleeping_branches=None,
        branch_connectivity=None,
    ):
        """Initialize Network.

        Parameters
        ----------
        box : Box
        branches : list, default []
        active_branches : list, default []
        sleeping_branches : list, default []
        branch_connectivity : array, default []

        Returns
        -------
        None.

        """

        self.box = box

        # all branches (to construct mesh): moving + sleeping + branches inside the tree
        self.branches = [] if branches is None else branches
        self.active_branches = [] if active_branches is None else active_branches  # moving branches (to extend)
        # branches without enough flux to move (may revive in the Poisson case)
        self.sleeping_branches = [] if sleeping_branches is None else sleeping_branches

        self.branch_connectivity = np.empty((0,2), dtype=int) if branch_connectivity is None else branch_connectivity

    def copy(self):
        """Return a deepcopy of the Network."""
        return copy.deepcopy(self)

    def height_and_length(self):
        """Return network height (max y coordinate) and total length of the branches."""
        ruler = 0
        height = 0
        for branch in self.branches:
            ruler = ruler + branch.length()
            height = np.max((height, np.max(branch.points[:, 1])))
        return height, ruler

    def add_connection(self, connection):
        """Add connection to self.branch_connectivity."""
        self.branch_connectivity = np.vstack(
                (self.branch_connectivity, connection))
        
    def reconnect(self, pde_solver, step):
        """Find potential anastomoses and reconnect."""       
        
        if "FreeFEM_ThickFingers" in type(pde_solver).__name__:
            reconnection_distance = pde_solver.finger_width + 5e-3
            reconnection_distance_bt = pde_solver.finger_width/2 + 5e-3 # 0.05*pde_solver.ds
        else:
            logger.warning("Reconnections for thin fingers? Check carefully!")
            reconnection_distance = 0.01*pde_solver.ds
            reconnection_distance_bt = 0.01*pde_solver.ds
            
        def index_branches():
            # branch.ID, branch ind,
            # starting pt. ind., ending pt. ind.,
            # starting x, starting y, ending x, ending y
            all_segments_branches = np.empty((0,8))
            for i, branch in enumerate(self.branches):
                n_points = len(branch.points)
                all_segments_branches = np.vstack(( all_segments_branches, \
                    np.column_stack( ( np.ones(n_points-1)*branch.ID,
                                    np.ones(n_points-1)*i,
                                    np.arange(n_points-1), np.arange(1, n_points),
                                    branch.points[:-1], branch.points[1:] )
                                    )
                    ) )
            return all_segments_branches
        
        def index_outlet():
            mask_outlet = (self.box.boundary_conditions==DIRICHLET_0_GLOB_FLUX) | \
                            (self.box.boundary_conditions==DIRICHLET_0) | \
                                (self.box.boundary_conditions==NEUMANN_0)
            inds_outlet = np.where(mask_outlet)[0]
            # starting pt. ind.,
            # starting x, starting y, ending x, ending y
            pts_outlet = self.box.points[self.box.connections[inds_outlet]]
            all_segments_outlet = np.column_stack( ( inds_outlet,
                                                    pts_outlet[:,0],
                                                    pts_outlet[:,1])
                                                    )
            return all_segments_outlet

        all_segments_branches = index_branches()
        all_segments_outlet = index_outlet()
        did_reconnect = False
        branches_to_iterate = self.active_branches.copy()
        for branch in branches_to_iterate:
            # BREAKTHROUGH
            min_distance, ind_min, is_pt_new, breakthrough_pt, _ = \
                            find_reconnection_point(branch.points[-1], \
                                                all_segments_outlet[...,1:3], \
                                                all_segments_outlet[...,3:], 
                                                too_close=1e-3)
                                
            if min_distance < reconnection_distance_bt:
                # decreasing step size while approaching BT:
                # remove False in if and uncomment pde_solver.ds=... in else below
                if False and pde_solver.ds>=1e-5:
                    pde_solver.ds = pde_solver.ds / 10
                    logger.info("! Branch %d is reaching the outlet ! ds = %s", branch.ID, pde_solver.ds)
                else:
                    did_reconnect = True
                    logger.info("! Branch %d broke through !", branch.ID)
                    # pde_solver.ds = 0.01

                    if is_pt_new:
                        ind_new_conn = int(all_segments_outlet[ind_min, 0])
                        ind_new_pt = ind_new_conn + 1
                        self.box.points = np.insert(self.box.points, \
                                                    ind_new_pt, \
                                                    breakthrough_pt, \
                                                    axis=0)
                        mask_temp = self.box.connections>=ind_new_pt
                        self.box.connections[mask_temp] = self.box.connections[mask_temp] + 1
                        new_conns = self.box.connections[ind_new_conn]
                        new_conns = [[new_conns[0], ind_new_pt], 
                                    [ind_new_pt, new_conns[1]]]
                        self.box.connections = np.vstack((self.box.connections[:ind_new_conn], 
                                                           new_conns,
                                                           self.box.connections[ind_new_conn+1:]))

                        self.box.boundary_conditions = \
                                    np.insert(self.box.boundary_conditions, \
                                                ind_new_conn, \
                                                self.box.boundary_conditions[ind_new_conn], \
                                                axis=0) 
                        
                        mask_temp = self.box.seeds_connectivity[:,0]>ind_new_pt
                        self.box.seeds_connectivity[mask_temp,0] = self.box.seeds_connectivity[mask_temp,0] + 1
                                  
                    branch.points = np.vstack( (branch.points, [breakthrough_pt]) )
                    branch.steps = np.append(branch.steps, [step+1])
                    self.active_branches.remove(branch)
                    self.add_connection([branch.ID, -1])

                    all_segments_outlet = index_outlet()
            
            # RECONNECTION TO OTHER BRANCHES
            elif len(self.branches)>1: # and branch.length() > 2*reconnection_distance:
                mask = np.ones(len(all_segments_branches), dtype=bool)        
                far_from_tip = sum(np.cumsum(np.flip(np.linalg.norm(branch.points[1:]-branch.points[:-1], axis=1)))>1.05*reconnection_distance)
                mask_branch = np.logical_and(all_segments_branches[:,0]==branch.ID, all_segments_branches[:,2]>=far_from_tip-1)
                mask[mask_branch] = False
                
                tip = branch.points[-1]
                    
                min_distance, ind_min, is_pt_new, reconnection_pt, _ = \
                                find_reconnection_point(tip, \
                                                    all_segments_branches[mask,4:6], \
                                                    all_segments_branches[mask,6:], 
                                                    too_close=1e-3)                    

                if min_distance < reconnection_distance:
                    did_reconnect = True
                    # to make more realistic reconnections for thick fingers we stretch tip further
                    if "FreeFEM_ThickFingers" in type(pde_solver).__name__:
                        dr1 = branch.points[-1] - branch.points[-2]
                        dr1 = dr1/np.linalg.norm(dr1) 
                        dr2 = reconnection_pt - branch.points[-1]
                        dr2 = dr2/np.linalg.norm(dr2)
                        dr = (dr1 + dr2) / 2
                        dr = dr / np.linalg.norm(dr) * pde_solver.finger_width/2
                        tip = branch.points[-1] + dr
                        _, ind_min, is_pt_new, reconnection_pt, _ = \
                                        find_reconnection_point(tip, \
                                                            all_segments_branches[mask,4:6], \
                                                            all_segments_branches[mask,6:], 
                                                            too_close=1e-3)
                    
                    # reconnect to a branch
                    branch2_id = int(all_segments_branches[mask][ind_min,0])
                    logger.info("! Branch %d reconnected to branch %d !", branch.ID, branch2_id)
                    
                    if "FreeFEM_ThickFingers" in type(pde_solver).__name__:
                        branch.points = np.vstack( (branch.points, [tip]) )
                        branch.steps = np.append(branch.steps, [step+1])
                    branch.points = np.vstack( (branch.points, [reconnection_pt]) )
                    branch.steps = np.append(branch.steps, [step+1])
                    self.active_branches.remove(branch)
                    self.add_connection([branch.ID, branch2_id])      
                    
                    if is_pt_new:
                        branch2_ind = int(all_segments_branches[mask][ind_min,1])
                        branch2 = self.branches[branch2_ind]
                        ind_pt = int(all_segments_branches[mask][ind_min, 2]) # ind_min is the index of the closest segment
                        branch2.points = np.insert(branch2.points, ind_pt+1, reconnection_pt, axis=0)
                        branch2.steps = np.insert(branch2.steps, ind_pt+1, branch2.steps[ind_pt+1], axis=0)

                        # update all_segments_branches (in case something else reconnects to the same branch)
                        all_segments_branches = all_segments_branches[ all_segments_branches[:,0]!=branch2_id]
                        n_points = len(branch2.points)
                        all_segments_branches = np.vstack(( all_segments_branches, \
                            np.column_stack( ( np.ones(n_points-1)*branch2.ID,
                                              np.ones(n_points-1)*branch2_ind,
                                              np.arange(n_points-1), np.arange(1, n_points),
                                              branch2.points[:-1], branch2.points[1:] )
                                            )
                            ) )
                    
                    all_segments_branches = index_branches()

                
        return did_reconnect
