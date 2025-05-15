"""Building blocks of the System.

Classes:
    Box
    Branch
    Network
"""

import numpy as np 
import copy

from reticuler.utilities.misc import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET_1, DIRICHLET_0, NEUMANN_0, NEUMANN_1, DIRICHLET_GLOB_FLUX
from reticuler.utilities.misc import cyl2cart, find_reconnection_point

class Branch:
    """A class of a single branch in a network.

    Attributes
    ----------
    ID : int
        Branch ID.
    points : array
        A 2-n array with xy coordinates of the points composing the branch.
        Chronological order (tip is the last point).
    steps : array
        A 1-n array with evolution steps at which corresponding points were added.
    BC : int, default 0
        The boundary condition on fingers, when solving the equations for the field.
        DIRICHLET_0 (u=0)
        DIRICHLET_1 (u=1)
    dR : array or 0, default 0
        A 1-2 array of tip progression ([dx, dy]).
    is_bifurcating : bool, default False
        A boolean condition if branch is bifurcating or not.
        (Based on the bifurcation_type and bifurcation_thresh from extender.)

    """

    def __init__(self, ID, points, steps, BC=DIRICHLET_0):
        """Initialize Branch.

        Parameters
        ----------
        ID : int
        points : array
        steps : array

        Returns
        -------
        None.

        """
        self.ID = ID

        self.points = points  # in order of creation
        self.steps = steps  # at which step of the evolution the point was added
        self.BC = BC # boundary condition
        
        self.dR = 0

    def extend(self, step=0):
        """Add a new point to ``self.points`` (progressed tip)."""
        if np.linalg.norm(self.dR) < 9e-5:
            print("! Extremely small dR, tip {} not extended but shifted !".format(self.ID))
            tip_versor = self.points[-1] - self.points[-2]
            tip_versor = tip_versor / np.linalg.norm(tip_versor)
            self.points[-1] = self.points[-1] + tip_versor * self.dR
        else:
            self.points = np.vstack((self.points, self.points[-1] + self.dR))
            self.steps = np.append(self.steps, step)

    def length(self):
        """Return length of the Branch."""
        return np.sum(np.linalg.norm(self.points[1:]-self.points[:-1], axis=1))

    def tip_angle(self):
        """Return the angle between the tip segment (last and penultimate point) and X axis."""
        point_penult = self.points[-2]
        point_last = self.points[-1]
        dx = point_last[0] - point_penult[0]
        dy = point_last[1] - point_penult[1]
        return np.arctan2(dy, dx)

    def points_steps(self):
        """Return a 3-n array of points and evolution steps when they were added."""
        return np.column_stack((self.points, self.steps))


class Box:
    """A class containing borders of the simulation domain.

    Attributes
    ----------
    points : array, default []
        A 2-n array with xy coordinates of the points composing the Box.
    connections : array, default []
        A 2-n array with connections between the ``points``.
    boundary_conditions : array, default []
        A 1-n array of boundary conditions \
        corresponding to links in ``connections`` list.
            - 1: absorbing BC (vanishing field)
            - 2: reflective BC (vanishing normal derivative)
            - 3: constant flux
    seeds_connectivity : array, default []
        A 2-n array of seeds connectivity.
            - 1st column: index in ``points``
            - 2nd column: outgoing branch ``ID`` 

    """

    def __init__(
        self, points=None, connections=None, boundary_conditions=None, seeds_connectivity=None
    ):
        """Initialize Box.

        Parameters
        ----------
        points : array, default []
        connections : array, default []
        boundary_conditions : array, default []
        seeds_connectivity : array, default []

        Returns
        -------
        None.

        """
        self.points = np.empty((0,2),dtype=float) if points is None else points
        self.connections = np.empty((0,2),dtype=int) if connections is None else connections
        self.boundary_conditions = np.empty((0,1),dtype=int) if boundary_conditions is None else boundary_conditions

        # 1st column: index on border
        # 2nd column: branch_id
        self.seeds_connectivity = [] if seeds_connectivity is None else seeds_connectivity
        
    def __add_points(self, points):
        self.points = np.vstack((self.points, points))

    def __add_connection(self, connections, boundary_conditions):
        self.connections = np.vstack((self.connections, connections))
        self.boundary_conditions = np.append(self.boundary_conditions, boundary_conditions)

    def connections_bc(self):
        """Return a 3-n array of connections and boundary conditions corresponding to them.
        (1st/2nd column - point indices, 3rd column - BC)
        """
        return np.column_stack((self.connections, self.boundary_conditions))

    def copy(self):
        """Return a deepcopy of the Box."""
        return copy.deepcopy(self)

    @classmethod
    def construct(cls, initial_condition=0, **kwargs_construct):
        """Construct a Box with given initial condition.

        Parameters
        ----------
        initial_condition : int, default 0
            IC = 0, 1, 2, 3. Rectangular box of dimensions ``width`` x ``height``,
            absorbing bottom wall, reflecting left and right, and:
                - IC = 0: constant flux on top (Laplacian case)
                - IC = 1: reflective top (Poissonian case)
                - IC = 2: PBC right and left wall + DIRICHLET_1 BC on top
                - IC = 3: as IC=0, but DIRICHLET_1 BC on top
            IC = 4, 5: jellyfish (an octant) with a trifork
                - IC = 4: Dirichlet on bottom and top, but rescaled such that global flux is constant
                - IC = 5: u=0 on top and Neumann on bottom
            IC = 6: leaf (semicircle with a single needle)
        kwargs_construct:
            IC = 0, 1, 2, 3
                seeds_x : array, default [0.5]
                    A 1-n array of x positions at the bottom boundary (y=0).
                initial_lengths : array, default [0.01]
                    A 1-n array of seeds initial lengths.
                    Length must match seeds_x or be equal to 1 
                    (then the same initial length will be set for all seeds).
                branch_BCs: array, default [DIRICHLET_0]
                    Boundary conditions on the branches.
                    Length must match seeds_x or be equal to 1 
                    (then the same initial length will be set for all seeds).
                height : float, default 50.0
                    Height of the rectangular system.
                width : float, default 2.0
                    Width of the rectangular system.
            IC = 6
                seeds_x : array, default [0]
                    A 1-n array of x positions at the bottom boundary (y=0).
                initial_lengths : array, default [0.4]
                    A 1-n array of seeds initial lengths.
                    Length must match seeds_x or be equal to 1 
                    (then the same initial length will be set for all seeds).
                radius : float, default 0.5
                    radius of the semicircle

        Returns
        -------
        box : Box
            An object of class Box.
        branches : list
            A list of objects of class Branch.


        """
        # Build a box
        box = cls()

        # Rectangular box of specified width and height
        if initial_condition == 0 or initial_condition == 1 or \
            initial_condition == 2 or initial_condition == 3:
            options_construct = {
                "seeds_x": [0.5],
                "initial_lengths": [0.01],
                "branch_BCs": [DIRICHLET_0],
                "height": 50.0,
                "width": 2.0,
            }
            options_construct.update(kwargs_construct)
            options_construct["seeds_x"]=np.array(options_construct["seeds_x"])
            if not len(options_construct["initial_lengths"])==len(options_construct["seeds_x"]):
                options_construct["initial_lengths"] = (
                    np.ones(len(options_construct["seeds_x"]))
                    * options_construct["initial_lengths"][0]
                )
            if not len(options_construct["branch_BCs"])==len(options_construct["seeds_x"]):
                options_construct["branch_BCs"] = (
                    np.ones(len(options_construct["seeds_x"]))
                    * options_construct["branch_BCs"][0]
                )
            options_construct["branch_BCs"]=np.array(options_construct["branch_BCs"])
            
            
            # right boundary
            box.__add_points(
                [
                    [options_construct["width"], 0],
                    [options_construct["width"], options_construct["height"]]
                ]
            )
            # seeds at the top boundary
            mask_seeds_from_outlet = options_construct["branch_BCs"]==DIRICHLET_1
            box.__add_points(
                np.vstack(
                    [
                        options_construct["seeds_x"][mask_seeds_from_outlet],
                        options_construct["height"]*\
                            np.ones(sum(mask_seeds_from_outlet)),
                    ]
                ).T
            )
            # left boundary
            box.__add_points(
                [
                    [0, options_construct["height"]],
                    [0, 0]
                ]
            )
            # seeds at the bottom boundary
            box.__add_points(
                np.vstack(
                    [
                        options_construct["seeds_x"][~mask_seeds_from_outlet],
                        np.zeros(sum(~mask_seeds_from_outlet)),
                    ]
                ).T
            )
            box.seeds_connectivity = np.column_stack(
                (
                    2 +
                    np.arange(sum(mask_seeds_from_outlet)),
                    np.arange(sum(mask_seeds_from_outlet)),
                )
            )
            box.seeds_connectivity = np.row_stack(
                (box.seeds_connectivity,
                np.column_stack(
                    (
                        len(box.points) - sum(~mask_seeds_from_outlet) +
                        np.arange(sum(~mask_seeds_from_outlet)),
                        sum(mask_seeds_from_outlet) + np.arange(sum(~mask_seeds_from_outlet)),
                    )
                )
            ))

            connections_to_add = np.vstack(
                [np.arange(len(box.points)), np.roll(
                    np.arange(len(box.points)), -1)]
            ).T
            box.__add_connection(
                connections_to_add,
                boundary_conditions=DIRICHLET_0 * \
                    np.ones(len(connections_to_add), dtype=int),
            )

            # right, left, top Neumann:
            box.boundary_conditions[0:3+sum(mask_seeds_from_outlet)] = NEUMANN_0
            # or top constant flux:
            if initial_condition == 0:
                box.boundary_conditions[1] = NEUMANN_1
            if initial_condition == 2:
                box.boundary_conditions[1] = DIRICHLET_1
                box.boundary_conditions[0] = RIGHT_WALL_PBC
                box.boundary_conditions[2] = LEFT_WALL_PBC
            if initial_condition == 3:
                box.boundary_conditions[1:2+sum(mask_seeds_from_outlet)] = DIRICHLET_1

            # Creating initial branches
            branches = []
            for i, x in enumerate(options_construct["seeds_x"]):
                BC = options_construct["branch_BCs"][i]
                if BC==DIRICHLET_0:                    
                    branch = Branch(
                            ID=i,
                            points=np.array(
                                [[x, 0], [x, options_construct["initial_lengths"][i]]]
                            ),
                            steps=np.array([0, 0]),
                        )
                elif BC==DIRICHLET_1:
                   branch = Branch(
                           ID=i,
                           points=np.array(
                               [[x, options_construct["height"]], \
                                [x, options_construct["height"]-options_construct["initial_lengths"][i]]]
                           ),
                           steps=np.array([0, 0]),
                           BC=BC
                       )
                branches.append(branch)
            
            active_branches = branches.copy()
            branch_connectivity = None
                
        # Jellyfish
        # make it smoother initially, so that it stays smooth when enlarging?        
        elif initial_condition == 4 or initial_condition == 5:
            angular_width = 2*np.pi / 8
            R_rim = 5 # mm
            R_stom = 0.45 * R_rim
            h0 = R_rim - R_stom
                
            # right boundary
            box.__add_points([cyl2cart(R_rim, angular_width/2, R_rim)])
            
            # stomach
            n_points_stomach = 48 # n_points_rim % 2 == 0
            box.__add_points(
                cyl2cart(R_stom, np.linspace(angular_width/2, -angular_width/2, n_points_stomach+1), R_rim)
            )
            # circular rim
            n_points_rim = 48 # n_points_rim % 8 == 0
            box.__add_points(
                cyl2cart(R_rim, np.linspace(-angular_width/2,0.99999*angular_width/2, n_points_rim+1), R_rim)
            )
            
            # seeds indices
            n0_rim = n_points_stomach+2
            box.seeds_connectivity = np.column_stack(
                (
                    [n0_rim+n_points_rim//2, 
                     n0_rim+n_points_rim//4, n0_rim+n_points_rim//4*3, 
                     n0_rim+n_points_rim//8, n0_rim+n_points_rim//8*3, 
                     n0_rim+n_points_rim//8*5, n0_rim+n_points_rim//8*7],
                    np.arange(7),
                )
            )
            # Connections and BCs
            connections_to_add = np.vstack(
                [np.arange(len(box.points)), np.roll(
                    np.arange(len(box.points)), -1)]
            ).T
            box.__add_connection(
                connections_to_add,
                boundary_conditions=DIRICHLET_1
                * np.ones(len(connections_to_add), dtype=int),
            )
            # right, left Neumann:
            box.boundary_conditions[0] = NEUMANN_0
            box.boundary_conditions[n_points_stomach+1] = NEUMANN_0
            # top DIRICHLET_GLOB_FLUX
            box.boundary_conditions[1:n_points_stomach+1] = DIRICHLET_GLOB_FLUX
            if initial_condition == 5:
                # bottom NEUMANN_1
                box.boundary_conditions[n_points_stomach+2:] = NEUMANN_1    
            
            # points_to_plot = box.points[box.connections]
            # for i, pts in enumerate(points_to_plot):
            #     plt.plot(*pts.T, '.-', color="{}".format(box.boundary_conditions[i]/5), ms=1, lw=5)
            
            # Creating initial branches
            branches = []
            active_branches = []
            # interradial canal
            n_inter = 42 # n_inter % 3 == 0
            branches.append(Branch(
                    ID=0,
                    points=cyl2cart(np.linspace(R_rim, R_stom, n_inter+1), 0, R_rim),
                    steps=np.zeros(n_inter+1),
                )
            )
            # trifork left
            n_trifork = 42
            t = np.linspace(angular_width/8, np.pi/2,n_trifork)
            r1 = np.sqrt( (2*R_rim*np.sin(angular_width/8))**2 - (2/3*h0*np.sin(angular_width/16))**2)/np.cos(angular_width/16)
            x = R_rim - r1 * np.cos(t)
            y = 2/3*h0 * np.sin(t)
            branches.append(Branch(
                    ID=1,
                    points=np.vstack((box.points[n0_rim+n_points_rim//4],np.vstack((x,y)).T)),
                    steps=np.zeros(n_trifork+1),
                )
            )
            # trifork right
            t = np.linspace(np.pi-angular_width/8, np.pi/2,n_trifork)
            r1 = np.sqrt( (2*R_rim*np.sin(angular_width/8))**2 - (2/3*h0*np.sin(angular_width/16))**2)/np.cos(angular_width/16)
            x = R_rim - r1 * np.cos(t)
            y = 2/3*h0 * np.sin(t)
            branches.append(Branch(
                    ID=2,
                    points=np.vstack((box.points[n0_rim+n_points_rim//4*3],np.vstack((x,y)).T)),
                    steps=np.zeros(n_trifork+1),
                )
            )
            # sprouts
            eps = np.array([0.013 , -0.012, 0.008, -0.011])
            # eps = np.random.rand(4)*0.01
            for i, theta in enumerate(np.arange(-3/8,3.1/8,1/4)*angular_width):
                branch = Branch(
                        ID=3+i,
                        points=cyl2cart(np.array([R_rim, R_rim-0.1]), theta+eps[i], R_rim),
                        steps=np.array([0, 0])
                    )
                branches.append(branch)       
                active_branches.append(branch)
                box.points[box.seeds_connectivity[3+i, 0]] = branch.points[0]
                
            branch_connectivity = np.array([[0,-1],[1,0],[2,0]])
        
        # Leaf semicircle
        elif initial_condition == 6:
            options_construct = {
                "seeds_x": [0],
                "initial_lengths": [0.4],
                "branch_BCs": [DIRICHLET_0],
                "radius": 0.5,
            }
            options_construct.update(kwargs_construct)
            options_construct["seeds_x"]=np.array(options_construct["seeds_x"])
            if not len(options_construct["initial_lengths"])==len(options_construct["seeds_x"]):
                options_construct["initial_lengths"] = (
                    np.ones(len(options_construct["seeds_x"]))
                    * options_construct["initial_lengths"][0]
                )
            if not len(options_construct["branch_BCs"])==len(options_construct["seeds_x"]):
                options_construct["branch_BCs"] = (
                    np.ones(len(options_construct["seeds_x"]))
                    * options_construct["branch_BCs"][0]
                )
            options_construct["branch_BCs"]=np.array(options_construct["branch_BCs"])        
        
            box = Box()

            angular_width = np.pi
            # circular rim
            n_points_rim = 100
            box.__add_points(
                np.vstack( cyl2cart(options_construct["radius"], np.linspace(np.pi-angular_width/2,np.pi+angular_width/2, n_points_rim+1), 0) )
            )
                        
            # seeds at the bottom boundary
            box.__add_points(
                np.vstack(
                    [
                        options_construct["seeds_x"],
                        np.zeros(len(options_construct["seeds_x"])),
                    ]
                ).T
            )
            
            # box.seeds_connectivity = [n_points_rim+len(options_construct["seeds_x"]), 0]
            n_seeds = len(options_construct["seeds_x"])
            box.seeds_connectivity = np.column_stack(
                    (
                        len(box.points) - n_seeds + np.arange(n_seeds),
                        np.arange(n_seeds),
                    )
                )

            # Connections and BCs
            connections_to_add = np.vstack(
                [np.arange(len(box.points)), np.roll(
                    np.arange(len(box.points)), -1)]
            ).T
            box.__add_connection(
                connections_to_add,
                boundary_conditions=DIRICHLET_1
                * np.ones(len(connections_to_add), dtype=int),
            )
            box.boundary_conditions[-1-n_seeds:] = NEUMANN_0

            # Creating initial branches
            branches = []
            active_branches = []
            for i, x in enumerate(options_construct["seeds_x"]):
                BC = options_construct["branch_BCs"][i]                   
                branch = Branch(
                        ID=i,
                        points=np.array(
                            [[x, 0], [x, options_construct["initial_lengths"][i]]]
                        ),
                        steps=np.array([0, 0]),
                        BC=BC
                    )
                branches.append(branch)
                active_branches.append(branch)            
            
            branch_connectivity = None
            
        return box, branches, active_branches, branch_connectivity


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
        
        if type(pde_solver).__name__ == "FreeFEM_ThickFingers":
            reconnection_distance = pde_solver.finger_width + 5e-3
            reconnection_distance_bt = pde_solver.finger_width/2 + 5e-3 # 0.05*pde_solver.ds
        else:
            print("Reconnections for thin fingers? Check carefully!")
            reconnection_distance = 0.01*pde_solver.ds
            reconnection_distance_bt = 0.01*pde_solver.ds
            
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
        
        mask_outlet = np.logical_or(self.box.boundary_conditions==DIRICHLET_GLOB_FLUX,
                                    self.box.boundary_conditions==DIRICHLET_0)
                                    # self.box.boundary_conditions==NEUMANN_1,)
        inds_outlet = np.where(mask_outlet)[0]
        # starting pt. ind.,
        # starting x, starting y, ending x, ending y
        pts_outlet = self.box.points[self.box.connections[inds_outlet]]
        all_segments_outlet = np.column_stack( ( inds_outlet,
                                                  pts_outlet[:,0],
                                                  pts_outlet[:,1])
                                                )
        
        did_reconnect = False
        for branch in self.active_branches:
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
                    print("! Branch {ID} is reaching the outlet ! ds = {ds}".format(ID=branch.ID, ds=pde_solver.ds))
                else:
                    did_reconnect = True
                    print("! Branch {ID} broke through !".format(ID=branch.ID))
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
            
            # RECONNECTION TO OTHER BRANCHES
            elif len(self.branches)>1 and branch.length() > 2*reconnection_distance:
                mask = np.ones(len(all_segments_branches), dtype=bool)        
                far_from_tip = sum(np.cumsum(np.flip(np.linalg.norm(branch.points[1:]-branch.points[:-1], axis=1)))>1.05*reconnection_distance)
                mask_branch = np.logical_and(all_segments_branches[:,0]==branch.ID, all_segments_branches[:,2]>=far_from_tip-1)
                mask[mask_branch] = False
                
                # if type(pde_solver).__name__ == "FreeFEM_ThickFingers":
                #     # dr = branch.points[-1] - branch.points[-2]
                #     # dr = dr/np.linalg.norm(dr) * pde_solver.finger_width/2 # !!!
                #     tip = branch.points[-1] # + dr
                # else:
                tip = branch.points[-1]
                    
                min_distance, ind_min, is_pt_new, reconnection_pt, _ = \
                                find_reconnection_point(tip, \
                                                    all_segments_branches[mask,4:6], \
                                                    all_segments_branches[mask,6:], 
                                                    too_close=1e-3)                    

                if min_distance < reconnection_distance:
                    did_reconnect = True
                    # to make more realistich reconnections we stretch tip further
                    dr = branch.points[-1] - branch.points[-2]
                    dr = dr/np.linalg.norm(dr) * pde_solver.finger_width/2 # !!!
                    tip = branch.points[-1] + dr
                    _, ind_min, is_pt_new, reconnection_pt, _ = \
                                    find_reconnection_point(tip, \
                                                        all_segments_branches[mask,4:6], \
                                                        all_segments_branches[mask,6:], 
                                                        too_close=1e-3) 
                    
                    # reconnect to a branch
                    branch2_id = int(all_segments_branches[mask][ind_min,0])
                    print("! Branch {ID} reconnected to branch {ID2} !".format(ID=branch.ID, ID2=branch2_id))
                    
                    if type(pde_solver).__name__ == "FreeFEM_ThickFingers":
                        branch.points = np.vstack( (branch.points, [tip]) )
                        branch.steps = np.append(branch.steps, [step+1])
                    branch.points = np.vstack( (branch.points, [reconnection_pt]) )
                    branch.steps = np.append(branch.steps, [step+1])
                    self.active_branches.remove(branch)
                    self.add_connection([branch.ID, branch2_id])      
                    
                    if is_pt_new:
                        branch2_ind = int(all_segments_branches[mask][ind_min,1])
                        branch2 = self.branches[branch2_ind]
                        ind_pt = int(all_segments_branches[mask][ind_min, 2])
                        branch2.points = np.insert(branch2.points, ind_pt+1, reconnection_pt, axis=0)
                        branch2.steps = np.insert(branch2.steps, ind_pt+1, branch2.steps[ind_pt], axis=0)

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
        return did_reconnect
                

    def move_tips(self, extender, step=0):
        """Move tips (with bifurcations and killing if is_testing==False)."""

        # shallow copy of active_branches (creates new list instance, but the elements are still the same)
        branches_to_iterate = self.active_branches.copy()
        for i, branch in enumerate(branches_to_iterate):
            if branch.dR.ndim==1:
                branch.extend(step)
            else:
                max_branch_id = len(self.branches) - 1
                for j, dR in enumerate(branch.dR):
                    points = np.array(
                        [branch.points[-1], branch.points[-1] + dR])
                    branch_new = Branch(
                        ID=max_branch_id + j + 1,
                        BC=branch.BC,
                        points=points,
                        steps=np.array([step - 1, step]),
                    )
                    self.branches.append(branch_new)
                    self.active_branches.append(branch_new)
                    self.add_connection([branch.ID, branch_new.ID])
                self.active_branches.remove(branch)
