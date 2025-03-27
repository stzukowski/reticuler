"""Classes for case-specific System manipulation.

Classes:
    Jellyfish
"""

import numpy as np

from reticuler.utilities.geometry import Branch
from reticuler.utilities.misc import DIRICHLET_1
from reticuler.utilities.misc import cyl2cart, cart2cyl, extend_radially, find_reconnection_point

class Jellyfish:
    """A class to handle jellyfish simulations. Includes global growth and adding new sprouts.
    """ 
    def __init__(
        self,
        radii,
        timescale=0.1,
        v_rim=1,
    ):
        """Initialize Jellyfish.

        Parameters
        ----------
        radii : array, default []
            How radius changed in time (corresponds to system.timestamps).
        timescale : float, default 0.1
            Factor to match the time when the first sprouts connect to stomachs.
        v_rim : float, default 1.0
            How fast jelly radius grow [mm/day].

        Returns
        -------
        None.

        """
        self.radii = np.array([0]) if radii is None else radii
        self.timescale = timescale
        self.v_rim = v_rim
        
    def morph(self, network, out_growth, step):
        # Global growth of the box (and network)
        
        # growth factor
        out_growth[0] = out_growth[0] * self.timescale 
        dt = out_growth[0]
        R_rim0 = self.radii[-1]
        beta = 1 + self.v_rim * dt / R_rim0 
        self.radii = np.append(self.radii, R_rim0*beta)
        
        # extend box and network
        network.box.points = extend_radially(network.box.points, R_rim0, beta)
        for branch in network.branches:
            branch.points = extend_radially(branch.points, R_rim0, beta)
            
        # check distances and add sprouts
        R_rim0 = self.radii[-1]
        canals_pos_ang = [-2*np.pi / 8 / 2, 2*np.pi / 8 / 2]
        for b in network.branches:
            r, t = cart2cyl(*b.points[0], R_rim0)
            canals_pos_ang.append(t)
        canals_pos_ang = np.sort(canals_pos_ang)
        distances_ang = np.diff(canals_pos_ang)
        mid_pos_ang = canals_pos_ang[:-1] + distances_ang/2

        max_branch_id = len(network.branches) - 1
        for i, theta in enumerate(mid_pos_ang[distances_ang*2*R_rim0>1.1]):
            print(f"Initiating new sprout at theta={theta/np.pi*180:.2f} deg.")
            branch = Branch(
                    ID=max_branch_id+i+1,
                    points=np.vstack( cyl2cart(np.array([R_rim0, R_rim0-0.1]), \
                                               theta, \
                                               R_rim0) ),
                    steps=np.array([step, step])
                )
            # +np.random.rand()*0.01
            network.branches.append(branch)       
            network.active_branches.append(branch)
            
            # place seed at the boundary
            seed = branch.points[0]
            _, ind_min, is_pt_new, _ , ind_min_end = find_reconnection_point(seed, \
                                                    network.box.points[:-1], \
                                                    network.box.points[1:], \
                                                    too_close=1e-3)
            if not is_pt_new:
                network.box.points[ind_min+ind_min_end] = seed
            else:
                network.box.points = np.insert(network.box.points, ind_min+1, seed, axis=0)
                network.box.connections = np.vstack((network.box.connections, [network.box.connections[-1,0]+1,0]))
                network.box.connections[-2,1] = network.box.connections[-1,0]
                network.box.boundary_conditions = np.append(network.box.boundary_conditions, DIRICHLET_1)

            network.box.seeds_connectivity[network.box.seeds_connectivity[:,0]>ind_min, 0] = network.box.seeds_connectivity[network.box.seeds_connectivity[:,0]>ind_min, 0] + 1
            network.box.seeds_connectivity = np.vstack((network.box.seeds_connectivity, [ind_min+1, branch.ID]))
        
        return out_growth