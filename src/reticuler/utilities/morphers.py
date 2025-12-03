"""Classes for case-specific System manipulation.

Classes:
    Jellyfish
"""

import numpy as np

from reticuler.utilities.geometry import Branch
from reticuler.utilities.misc import cyl2cart, cart2cyl, extend_radially, find_reconnection_point
from reticuler.utilities.misc import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET_1, DIRICHLET_0, NEUMANN_0, NEUMANN_1, DIRICHLET_GLOB_FLUX

class Jellyfish:
    """A class to handle jellyfish simulations. Includes global growth and adding new sprouts.

    Attributes
    ----------
    radii : array, default [0]
        How radius changed in time (corresponds to system.timestamps).       
    timescale : float, default 1
        1) (earlier) Factor to match the time when the first sprout connects to stomach.
        dt = dt * timescale
        2) (now) timescale = pde_solver.ds / v_sprout_desired
        Assuming that the fastest sprout grows at constant speed v_sprout_desired => dt = timescale = ds / v_sprout_desired
    v_rim : float, default 1.4
        Jellyfish radius growth rate [mm/day].
    sprouting_thresh : float, default 1.5
        Threshold for inserting new sprouts [mm].
    sprouting_stochastic_factor : float, default 0.2
        Factor to randomize the position of new sprouts [mm].        

    """ 
    def __init__(
        self,
        radii=None,
        timescale=1,
        v_rim=1.4,
        sprouting_thresh=1.5,
        sprouting_stochastic_factor=0.25,
    ):
        """Initialize Jellyfish.

        Parameters
        ----------
        radii : array, default [0]
        timescale : float, default 1
        v_rim : float, default 1.4
        sprouting_thresh : float, default 1.5
        sprouting_stochastic_factor : float, default 0.2        

        Returns
        -------
        None.

        """
        self.sprouting_thresh = sprouting_thresh
        self.sprouting_stochastic_factor = sprouting_stochastic_factor
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
        out_growth[0] = self.timescale # * out_growth[0]
        dt = out_growth[0]
        R_rim0 = self.radii[-1]
        beta = 1 + self.v_rim * dt / R_rim0 

        self.radii, R_rim0, distances_ang, mid_pos_ang = extend_box_check_distances()

        if len(network.active_branches)==0 and \
          not any(distances_ang*R_rim0>=self.sprouting_thresh):
            print("No sprouts and no space for new ones. Additional jellyfish growth.")
            R_rim1 = (self.sprouting_thresh+1e-4) / max(distances_ang)
            beta = R_rim1 / R_rim0
            out_growth[0] = [dt, (R_rim1 - R_rim0)/self.v_rim]
            step = step + 1
            self.radii, R_rim0, distances_ang, mid_pos_ang = extend_box_check_distances()
            
        max_branch_id = len(network.branches) - 1
        for i, theta in enumerate(mid_pos_ang[distances_ang*R_rim0>=self.sprouting_thresh]):
            theta = theta + np.random.uniform(low=-1, high=1)*self.sprouting_stochastic_factor/R_rim0
            print(f"Initiating new sprout at theta={theta/np.pi*180:.2f} deg.")
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
                                                    network.box.points[:-1], \
                                                    network.box.points[1:], \
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
        
        return out_growth
  
# import matplotlib.pyplot as plt
# fig2,ax2 = plt.subplots()
# np.savez_compressed() to save box_history?
class Leaf:
    """A class to handle evolution of the boundary.

    Attributes
    ----------
    box_history : list, default []
        History of the box at each step.
    pts_min_separation : float, default 0.015
        Minimum separation between boundary points.
    pts_max_separation : float, default 0.03
        Maximum separation between boundary points.
    v_rim : float, default 1.0
        How fast the rim grows.
    response_func : str, default "linear"
        Function to map fluxes to boundary growth velocity.
    response_func_params : dict, default {}
        Parameters for the response function.
    """ 
    def __init__(
        self,
        box_history=[],
        pts_min_separation=0.015,
        pts_max_separation=0.03,
        v_rim=1,
        response_func="linear",
        response_func_params={},
    ):
        """Initialize Leaf.

        Parameters
        ----------
        box_history : list, default []
        v_rim : float, default 1.0
        pts_min_separation : float, default 0.015
        pts_max_separation : float, default 0.03
        v_rim : float, default 1.0
        response_func : str, default "linear"
        response_func_params : dict, default {}

        Returns
        -------
        None.

        """
        self.box_history = box_history
        self.v_rim = v_rim
        self.pts_min_separation = pts_min_separation
        self.pts_max_separation = pts_max_separation
        if response_func == "linear":
            self.response_func = self.linear
        if response_func == "sigmoid":
            self.response_func = self.sigmoid
        self.response_func_params = response_func_params
    
    def linear(self, fluxes):
            return fluxes
    def sigmoid(self, fluxes, shift=0.5, rate=10, height=1):
        return height / ( 1 + np.exp((fluxes-shift)*rate))

    def morph(self, network, out_growth, step):
        
        ###### IMPORT FLUXES ######
        top_xy_flux = out_growth[1]
        top_xy_flux = np.vstack((top_xy_flux[1],top_xy_flux[::2]))
        if network.box.initial_condition==350:
            top_xy_flux = top_xy_flux[:-1]
        x = top_xy_flux[:,0]
        y = top_xy_flux[:,1]
        fluxes = top_xy_flux[:,2]
        dt = out_growth[0]
        
        # VELOCITY
        velocity = self.response_func(fluxes, **self.response_func_params)
        # plt.pause(0.01)
        # velocity = fluxes.max()/(1+np.exp((np.quantile(fluxes0,0.6)-fluxes0)*3))
        # ax2.clear()
        # ax2.plot(np.arctan2(y,x),fluxes, '.-', ms=5)
        # ax2.plot(np.arctan2(y,x),velocity, '.-', ms=5)
        # plt.pause(0.01)

        ####### PUSH THE BOUNDARY ######
        vx = np.diff(x,prepend=2*x[0]-x[1],append=2*x[-1]-x[-2]) # warunki na brzegach = symetria względem ostatniego punktu
        vy = np.diff(y,prepend=2*y[0]-y[1],append=2*y[-1]-y[-2])
        if network.box.initial_condition==300:
            vy = np.diff(y,prepend=y[1],append=y[-2]) # warunki na brzegach = odbicie względem osi pionowej (tylko dla prostokątów)
        if network.box.initial_condition==350:
            vx = np.diff(x,prepend=x[-1],append=x[0]) # warunki na brzegach = cykliczne (tylko dla pełnego koła)
            vy = np.diff(y,prepend=y[-1],append=y[0])
        alfa = (np.arctan2(-vy[:-1],-vx[:-1])+np.arctan2(vy[1:],vx[1:]))/2 # kąt nachylenia dwusiecznej (między 1->0 a 1->2)
        s = self.v_rim*dt
        sx = s*velocity*np.cos(alfa) # definicja dwusiecznej i wartość przesunięcia z fluxów
        sy = s*velocity*np.sin(alfa)
        x += (2*(vx[1:]*sy<vy[1:]*sx)-1)*sx # przesuwanie punktów (zmiana znaku nierówności zmieni zwrot)
        y += (2*(vx[1:]*sy<vy[1:]*sx)-1)*sy
        
        ###### CONTROL POINT DENSITY ######
        points = np.array([x, y]).T
        if network.box.initial_condition==350: # Circular case
            points = np.vstack((points, points[0]))
        processed_points = [points[0]]
        for i in range(1, len(points)):
            last_kept_point = processed_points[-1]
            current_point = points[i]
            distance = np.linalg.norm(current_point - last_kept_point)
            # Too far apart -> Insert interpolated points
            if distance > self.pts_max_separation:
                # Calculate how many segments of self.pts_max_separation length fit between the points
                num_segments = int(np.ceil(distance / self.pts_max_separation))
                # Use linear interpolation to generate the intermediate points
                t_values = np.linspace(0, 1, num_segments + 1)
                interpolated_points = (1 - t_values[:, np.newaxis]) * last_kept_point + t_values[:, np.newaxis] * current_point
                # Add all new points to the list, except the first (which is last_kept_point)
                processed_points.extend(interpolated_points[1:])
            # Accepted separation range -> Keep the point
            elif distance >= self.pts_min_separation:
                processed_points.append(current_point)
            # If the last two points are too close, replace the penultimate point with the last
            elif i==len(points)-1:
                processed_points[-1] = current_point
        if network.box.initial_condition==350: # Circular case
            processed_points.pop(-1)
                    
        processed_points = np.array(processed_points)
        x, y = processed_points[:,0], processed_points[:,1]
            
        ###### UPDATE BOX ######
        n_seeds = np.sum(network.box.boundary_conditions!=DIRICHLET_1)-1
        if network.box.initial_condition==300:
            n_seeds-=2
            network.box.points = np.vstack(( network.box.points[0], np.stack((x,y)).T, network.box.points[-1-n_seeds:] ))
            network.box.points[1,0] = network.box.points[0,0]
            network.box.points[-2-n_seeds,0] = 0
        if network.box.initial_condition==301:
            network.box.points = np.vstack(( np.stack((x,y)).T, network.box.points[-n_seeds:] ))
            network.box.points[0,1] = 0
            network.box.points[-1-n_seeds,1] = 0
        if network.box.initial_condition==350:
            network.box.points = np.stack((x,y)).T
        if network.box.initial_condition==351:
            aw=np.atan2(*network.box.points[0])*2
            network.box.points = np.vstack(( np.stack((x,y)).T, [0,0] ))
            x=network.box.points[0,0]
            y=network.box.points[0,1]
            network.box.points[0,0] = ((1-np.cos(aw))*x + np.sin(aw)*y)/2
            network.box.points[0,1] = ((1+np.cos(aw))*y + np.sin(aw)*x)/2
            x=network.box.points[-2,0]
            y=network.box.points[-2,1]
            network.box.points[-2,0] = ((1-np.cos(aw))*x - np.sin(aw)*y)/2
            network.box.points[-2,1] = ((1+np.cos(aw))*y - np.sin(aw)*x)/2
        network.box.seeds_connectivity = np.column_stack(
                    (
                        len(network.box.points) - n_seeds + np.arange(n_seeds),
                        np.arange(n_seeds),
                    )
                )
        network.box.connections = np.vstack(
            [np.arange(len(network.box.points)), np.roll(
                np.arange(len(network.box.points)), -1)]
        ).T
        network.box.boundary_conditions = DIRICHLET_1 * np.ones(len(network.box.connections), dtype=int)
        if network.box.initial_condition==300:
            network.box.boundary_conditions[0] = NEUMANN_0
            network.box.boundary_conditions[-2-n_seeds] = NEUMANN_0
            network.box.boundary_conditions[-1-n_seeds:] = DIRICHLET_0
        if network.box.initial_condition==301 or (network.box.initial_condition==351):
            network.box.boundary_conditions[-1-n_seeds:] = NEUMANN_0
        self.box_history.append(network.box.copy())
        
        return out_growth