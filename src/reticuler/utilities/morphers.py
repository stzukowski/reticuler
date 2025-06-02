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
            How fast jelly radius grows [mm/day].

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
                network.box.boundary_conditions = np.append(network.box.boundary_conditions, network.box.boundary_conditions[-1])

            network.box.seeds_connectivity[network.box.seeds_connectivity[:,0]>ind_min, 0] = network.box.seeds_connectivity[network.box.seeds_connectivity[:,0]>ind_min, 0] + 1
            network.box.seeds_connectivity = np.vstack((network.box.seeds_connectivity, [ind_min+1, branch.ID]))
        
        return out_growth
  
# import matplotlib.pyplot as plt
# fig2,ax2 = plt.subplots()
class Leaf:
    """A class to handle evolution of the boundary.
    """ 
    def __init__(
        self,
        box_history=[],
        v_rim=1,
    ):
        """Initialize Leaf.

        Parameters
        ----------
        box_init : Box
            An object of class Box.
        v_rim : float, default 1.0
            How fast the rim grows [mm/day].

        Returns
        -------
        None.

        """
        self.v_rim = v_rim
        self.box_history = box_history
    
    def morph(self, network, out_growth, step):
        # Boundary dynamics
        top_xy_flux = out_growth[1]
        top_xy_flux = np.vstack((top_xy_flux[1],top_xy_flux[::2]))
        if network.box.initial_condition==7:
            top_xy_flux = top_xy_flux[:-1]
        x = top_xy_flux[:,0]
        y = top_xy_flux[:,1]
        fluxes = top_xy_flux[:,2];
        
        # SIGMOIDA
        # fluxes0=fluxes;
        # fluxes = fluxes0.max()/(1+np.exp((np.quantile(fluxes0,0.6)-fluxes0)*3))
        # # fluxes0.max()/(1+np.exp((np.mean(fluxes0)-fluxes0)*100))
        # ax2.clear()
        # ax2.plot(np.arctan2(y,x),fluxes0, '.-', ms=5)
        # ax2.plot(np.arctan2(y,x),fluxes, '.-', ms=5)
        # plt.pause(0.01)

        # PUSH THE BOUNDARY
        s=self.v_rim*out_growth[0] # mnożnik fluxów
        vx=np.diff(x,prepend=2*x[0]-x[1],append=2*x[-1]-x[-2]) # warunki na brzegach = lustro względem ostatniego punktu (dla semicircle/rectangle)
        vy=np.diff(y,prepend=2*y[0]-y[1],append=2*y[-1]-y[-2])
        if network.box.initial_condition==7:
            vx=np.diff(x,prepend=x[-1],append=x[0]) # warunki na brzegach = cykliczne
            vy=np.diff(y,prepend=y[-1],append=y[0])
        alfa=(np.arctan2(-vy[:-1],-vx[:-1])+np.arctan2(vy[1:],vx[1:]))/2 # kąt nachylenia dwusiecznej (między 1->0 a 1->2)
        sx=s*fluxes*np.cos(alfa) # definicja dwusiecznej i wartość przesunięcia z fluxów
        sy=s*fluxes*np.sin(alfa)
        x+=(2*(vx[1:]*sy<vy[1:]*sx)-1)*sx # przesuwanie punktów (zmiana znaku nierówności zmieni zwrot)
        y+=(2*(vx[1:]*sy<vy[1:]*sx)-1)*sy
        
        min=0
        max=0.02
        #REMOVE POINTS
        if min>0:
            tooclose=np.logical_or(vx[1:]**2+vy[1:]**2<min**2, vx[:-1]**2+vy[:-1]**2<min**2) #wektor poprzedni lub następny za krótki
            sharp=np.abs(np.arctan2(vy[:-1],vx[:-1])-np.arctan2(vy[1:],vx[1:]))>np.pi/2 #wykrywacz dzióbków (kąt między 0->1 a 1->2 za duży)
            par=np.array(range(x.size))%2 #0 dla parzystych, 1 dla nieparzystych, zrównanie długości par do x
            x=np.delete(x, np.logical_or(np.logical_and(par, tooclose),sharp)) #usuwanie co drugiego punktu oraz dzióbków
            y=np.delete(y, np.logical_or(np.logical_and(par, tooclose),sharp))
        #ADD POINTS
        if max!=None:
            midx=(x[:-1]+x[1:])/2 #środki odcinków
            midy=(y[:-1]+y[1:])/2
            toofar=np.append(False,np.diff(x)**2+np.diff(y)**2>max**2) #wektor następny za długi
            x=np.insert(x,toofar,midx[toofar[1:]]) #wstawianie punktów w odpowiednich miejscach
            y=np.insert(y,toofar,midy[toofar[1:]])
        
        # UPDATE BOX
        n_seeds = np.sum(network.box.boundary_conditions!=DIRICHLET_1)-1
        if network.box.initial_condition==8:
            n_seeds-=2
            network.box.points = np.vstack(( network.box.points[0], np.stack((x,y)).T, network.box.points[-1-n_seeds:] ))
            network.box.points[1,0] = network.box.points[0,0]
            network.box.points[-2-n_seeds,0] = 0
        if network.box.initial_condition==6:
            network.box.points = np.vstack(( np.stack((x,y)).T, network.box.points[-n_seeds:] ))
            network.box.points[0,1] = 0
            network.box.points[-1-n_seeds,1] = 0
        if network.box.initial_condition==7:
            network.box.points = np.stack((x,y)).T
            
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
        if network.box.initial_condition==8:
            network.box.boundary_conditions[0] = NEUMANN_0
            network.box.boundary_conditions[-2-n_seeds] = NEUMANN_0
            network.box.boundary_conditions[-1-n_seeds:] = DIRICHLET_0
        if network.box.initial_condition==6:
            network.box.boundary_conditions[-1-n_seeds:] = NEUMANN_0
            
        self.box_history.append(network.box.copy())
        
        return out_growth