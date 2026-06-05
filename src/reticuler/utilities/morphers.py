"""Classes for case-specific System manipulation.

Classes:
    Jellyfish
"""

import numpy as np

from reticuler.utilities.geometry import Branch
from reticuler.utilities.misc import cyl2cart, cart2cyl, extend_radially, find_reconnection_point, sigmoid
from reticuler.utilities.misc import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET_1, DIRICHLET_0, NEUMANN_0, NEUMANN_1, DIRICHLET_0_GLOB_FLUX

class Jellyfish:
    """A class to handle jellyfish simulations. Includes global growth and adding new sprouts.

    Attributes
    ----------
    radii : array, default [0]
        How radius changed in time (corresponds to system.timestamps).       
    timescale : float, default 1
        1) (v0, v2now) Factor to match the time when the first sprout connects to stomach.
        dt = dt * timescale
        2) (v1) timescale = pde_solver.ds / v_sprout_desired
        Assuming that the fastest sprout grows at constant speed v_sprout_desired 
        => dt = timescale = ds / v_sprout_desired
    v_rim : float, default 1.3
        Jellyfish radius growth rate [mm/day].
    sprouting_stochastic_shift : float, default 0.2
        Factor to randomize the position of new sprouts.  
    sprouting_thresh : float, default 1.5
        Threshold for inserting new sprouts [mm].
    sprouting_sig_rate : float, default 0.15
        Sigmoid rate for sprouting probability.
    sprouting_sig_h : float, default 10
        Maximum sprouting hazard.      

    """ 
    def __init__(
        self,
        radii=None,
        timescale=1,
        v_rim=1.3,
        sprouting_stochastic_shift=0.25,
        sprouting_thresh=1.5,
        sprouting_sig_rate=0.15,
        sprouting_sig_h=10,
    ):
        """Initialize Jellyfish.

        Parameters
        ----------
        radii : array, default [0]
        timescale : float, default 1
        v_rim : float, default 1.3 [mm/day]
        sprouting_stochastic_shift : float, default 0.25
        sprouting_thresh : float, default 1.5
        sprouting_sig_rate : float, default 0.15
        sprouting_sig_h : float, default 10

        Returns
        -------
        None.

        """
        self.sprouting_stochastic_shift = sprouting_stochastic_shift
        self.sprouting_thresh = sprouting_thresh
        self.sprouting_sig_rate = sprouting_sig_rate
        self.sprouting_sig_h = sprouting_sig_h
        self.radii = np.array([0]) if radii is None else radii
        self.timescale = timescale
        self.v_rim = v_rim
        
    def morph(self, network, out_growth, step):
        # Global growth of the box (and network)
        
        def extend_box_check_distances():
            # extend box and network
            new_radii = np.append(self.radii, R_rim0*beta)
            network.box.points[:] = extend_radially(network.box.points, R_rim0, beta)
            for branch in network.branches:
                branch.points[:] = extend_radially(branch.points, R_rim0, beta)

            # check distances and add sprouts
            R_rim01 = new_radii[-1]
            canals_pos_ang = [-2*np.pi / 8 / 2, 2*np.pi / 8 / 2]
            for b in network.branches:
                _, t = cart2cyl(*b.points[0], R_rim01)
                canals_pos_ang.append(t)
            canals_pos_ang = np.sort(canals_pos_ang)
            distances_ang = np.diff(canals_pos_ang)
            mid_pos_ang = canals_pos_ang[:-1] + distances_ang/2
            
            return new_radii, R_rim01, distances_ang, mid_pos_ang

        # growth factor
        dt = self.timescale * out_growth[0]
        out_growth[0] = dt # to export properly
        R_rim0 = self.radii[-1]
        beta = 1 + self.v_rim * dt / R_rim0 

        self.radii, R_rim0, distances_ang, mid_pos_ang = extend_box_check_distances()
        # updated sprouting_thresh (L0)!
        # self.sprouting_thresh = 2.9+0.03*R_rim0
        self.sprouting_thresh = 0.04*R_rim0
        
        ds = distances_ang * (self.radii[-1] - self.radii[-2])
        Ps = ds * sigmoid(distances_ang*R_rim0, \
                            sig_shift=self.sprouting_thresh, \
                            sig_rate=self.sprouting_sig_rate, sig_h=self.sprouting_sig_h)
        is_triggered = np.random.rand(len(Ps)) < Ps
        # is_triggered = distances_ang*R_rim0>=self.sprouting_thresh
        # canals_pos_ang.extend(mid_pos_ang[is_triggered])

        if len(network.active_branches)==0 and \
          not any(is_triggered):
            if self.sprouting_thresh < max(distances_ang) * R_rim0:
                dr = self.radii[-1] - self.radii[-2]
            else:
                dr = 1/20 * (self.sprouting_thresh / max(distances_ang) - R_rim0)
            ds = distances_ang * dr
            while not any(is_triggered):
                R_rim1 = R_rim0 + dr
                beta = R_rim1 / R_rim0
                new_radii, R_rim0, distances_ang, mid_pos_ang = extend_box_check_distances()
                # updated sprouting_thresh (L0) and width!
                # self.sprouting_thresh = 2.9+0.03*R_rim0
                self.sprouting_thresh = 0.04*R_rim0
                Ps = ds * sigmoid(distances_ang*R_rim0, \
                            sig_shift=self.sprouting_thresh, \
                            sig_rate=self.sprouting_sig_rate, sig_h=self.sprouting_sig_h)
                is_triggered = np.random.rand(len(Ps)) < Ps
            self.radii = new_radii
            out_growth[0] = [dt, (self.radii[-1] - self.radii[-2])/self.v_rim]
            step = step + 1
            print("No sprouts and no space for new ones. Additional jellyfish growth.")
        
        max_branch_id = len(network.branches) - 1
        for i, theta in enumerate(mid_pos_ang[is_triggered]):
            theta = theta + np.random.uniform(low=-1, high=1)*self.sprouting_stochastic_shift/R_rim0
            branch = Branch(
                    ID=max_branch_id+i+1,
                    points=np.vstack( cyl2cart(np.array([R_rim0, R_rim0-0.075]), \
                                               theta, \
                                               R_rim0) ),
                    steps=np.array([step, step])
                )
            network.branches.append(branch)
            network.active_branches.append(branch)
            
            # place seed at the boundary
            seed = branch.points[0]
            _, ind_min, is_pt_new, _ , ind_min_end = find_reconnection_point(seed, \
                                                    network.box.points, \
                                                    np.roll(network.box.points, -1, axis=0), \
                                                    too_close=1e-3)
            if not is_pt_new:
                network.box.points[ind_min+ind_min_end] = seed
            else:
                network.box.points = np.insert(network.box.points, ind_min+1, seed, axis=0)
                network.box.connections = np.vstack((network.box.connections, [network.box.connections[-1,0]+1,0]))
                network.box.connections[-2,1] = network.box.connections[-1,0]
                network.box.boundary_conditions = np.append(network.box.boundary_conditions, network.box.boundary_conditions[-1])

            network.box.seeds_connectivity[network.box.seeds_connectivity[:,0]>ind_min, 0] = network.box.seeds_connectivity[network.box.seeds_connectivity[:,0]>ind_min, 0] + 1
            network.box.seeds_connectivity = np.vstack((network.box.seeds_connectivity, [ind_min+1, branch.ID]))
        
        if any(is_triggered):
            thetas = [f"{t:.3f}" for t in mid_pos_ang[is_triggered]/np.pi*180]
            print(f"Initiating {sum(is_triggered)} new sprouts at \n theta={', '.join(thetas)} deg.")
        print(f"Radius: {self.radii[-1]:.3f} mm")
        
        return out_growth
