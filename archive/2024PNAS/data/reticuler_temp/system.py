"""Handle network simulations.

Classes:
    Box
    Branch
    Network
    System
    
"""

import numpy as np
import datetime
import time
import copy
import json
import importlib.metadata

from reticuler.extending_kernels import extenders, pde_solvers

# Labels for boundary conditions
DIRICHLET = 1
NEUMANN = 2
CONSTANT_FLUX = 3
DIRICHLET_OUTLET = 4
RIGHT_WALL_PBC = 999
LEFT_WALL_PBC = 998

def find_reconnection_point(p, a, b, too_close=0.1):
    """Cartesian distance from point to line segment
    https://stackoverflow.com/a/58781995

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    
    # closest points on the segments
    # we don't put new node if it's too close to the old ones
    # e = (s>t).reshape(3,1) * (a - ((s<-too_close)*s).reshape(3,1)*d) + \
    #     (s<t).reshape(3,1) * (b + ((t<-too_close)*t).reshape(3,1)*d)

    distances = np.hypot(h, c)
    ind_min = np.argmin(distances)
    
    # we don't put new node if it's too close to the old ones
    s1 = s[ind_min]
    t1 = t[ind_min]
    if s1>t1:
        is_pt_new = s1<-too_close
        breakthrough_pt = a[ind_min] - is_pt_new*s1*d[ind_min]
    else:
        is_pt_new = t1<-too_close
        breakthrough_pt = b[ind_min] + is_pt_new*t1*d[ind_min]
    
    return distances[ind_min], ind_min, is_pt_new, breakthrough_pt


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64): 
            return int(obj)
        return json.JSONEncoder.default(self, obj)


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
        0 - Dirichlet (phi=0)
        4 - Dirichlet outlet (phi=1)
    dR : array or 0, default 0
        A 1-2 array of tip progression ([dx, dy]).
    is_bifurcating : bool, default False
        A boolean condition if branch is bifurcating or not.
        (Based on the bifurcation_type and bifurcation_thresh from extender.)

    """

    def __init__(self, ID, points, steps, BC=DIRICHLET):
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
                - IC = 2: as IC=0 + PBC right and left wall
                - IC = 3: as IC=0, but Dirichlet BC on top
                - IC = 4: jellyfish (an octant) with a trifork
        kwargs_construct:
            IC = 0, 1, 2, 3
                seeds_x : array, default [0.5]
                    A 1-n array of x positions at the bottom boundary (y=0).
                initial_lengths : array, default [0.01]
                    A 1-n array of seeds initial lengths.
                    Length must match seeds_x or be equal to 1 
                    (then the same initial length will be set for all seeds).
                height : float, default 50.0
                    Height of the rectangular system.
                width : float, default 2.0
                    Width of the rectangular system.

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
                "branch_BCs": [DIRICHLET],
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
            mask_seeds_from_outlet = options_construct["branch_BCs"]==DIRICHLET_OUTLET
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
                boundary_conditions=DIRICHLET
                * np.ones(len(connections_to_add), dtype=int),
            )

            # right, left, top Neumann:
            box.boundary_conditions[0:3+sum(mask_seeds_from_outlet)] = NEUMANN
            # or top constant flux:
            if initial_condition == 0:
                box.boundary_conditions[1] = CONSTANT_FLUX
            if initial_condition == 2:
                box.boundary_conditions[1] = DIRICHLET_OUTLET
                box.boundary_conditions[0] = RIGHT_WALL_PBC
                box.boundary_conditions[2] = LEFT_WALL_PBC
            if initial_condition == 3:
                box.boundary_conditions[1:2+sum(mask_seeds_from_outlet)] = DIRICHLET_OUTLET

            # Creating initial branches
            branches = []
            for i, x in enumerate(options_construct["seeds_x"]):
                BC = options_construct["branch_BCs"][i]
                if BC==DIRICHLET:                    
                    branch = Branch(
                            ID=i,
                            points=np.array(
                                [[x, 0], [x, options_construct["initial_lengths"][i]]]
                            ),
                            steps=np.array([0, 0]),
                        )
                elif BC==DIRICHLET_OUTLET:
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
        elif initial_condition == 4:
            def cyl2cart(r, theta):
                # theta measured from the negative Y axis
                return [R0+r*np.sin(theta), R0-r*np.cos(theta)]
            angular_width = 2*np.pi / 8
            R0 = 5 # mm
            r0 = 0.4 * R0
            h0 = R0 - r0
                
            # right boundary
            box.__add_points([cyl2cart(R0, angular_width/2)])
            
            # stomach
            n_points_stomach = 24 # n_points_rim % 2 == 0
            box.__add_points(
                np.vstack( cyl2cart(r0, np.linspace(angular_width/2, -angular_width/2, n_points_stomach+1)) ).T
            )
            # circular rim
            n_points_rim = 24 # n_points_rim % 8 == 0
            box.__add_points(
                np.vstack( cyl2cart(R0, np.linspace(-angular_width/2,0.99999*angular_width/2, n_points_rim+1)) ).T
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
                boundary_conditions=DIRICHLET
                * np.ones(len(connections_to_add), dtype=int),
            )
            # right, left Neumann:
            box.boundary_conditions[0] = NEUMANN
            box.boundary_conditions[n_points_stomach+1] = NEUMANN
            # top Dirichlet
            box.boundary_conditions[1:n_points_stomach+1] = DIRICHLET_OUTLET
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
                    points=np.vstack(cyl2cart(np.linspace(R0, r0, n_inter+1), 0)).T,
                    steps=np.zeros(n_inter+1),
                )
            )
            # trifork left
            n_trifork = 42
            t = np.linspace(angular_width/8, np.pi/2,n_trifork)
            r1 = np.sqrt( (2*R0*np.sin(angular_width/8))**2 - (2/3*h0*np.sin(angular_width/16))**2)/np.cos(angular_width/16)
            x = R0 - r1 * np.cos(t)
            y = 2/3*h0 * np.sin(t)
            branches.append(Branch(
                    ID=1,
                    points=np.vstack((box.points[n0_rim+n_points_rim//4],np.vstack((x,y)).T)),
                    steps=np.zeros(n_trifork+1),
                )
            )
            # trifork right
            t = np.linspace(np.pi-angular_width/8, np.pi/2,n_trifork)
            r1 = np.sqrt( (2*R0*np.sin(angular_width/8))**2 - (2/3*h0*np.sin(angular_width/16))**2)/np.cos(angular_width/16)
            x = R0 - r1 * np.cos(t)
            y = 2/3*h0 * np.sin(t)
            branches.append(Branch(
                    ID=2,
                    points=np.vstack((box.points[n0_rim+n_points_rim//4*3],np.vstack((x,y)).T)),
                    steps=np.zeros(n_trifork+1),
                )
            )
            # sprouts
            for i, theta in enumerate(np.arange(-3/8,3.1/8,1/4)*angular_width):
                branch = Branch(
                        ID=3+i,
                        points=np.vstack(cyl2cart(np.array([R0, R0-0.1]), theta)).T,
                        steps=np.array([0, 0])
                    )
                branches.append(branch)       
                active_branches.append(branch)
                
            branch_connectivity = np.array([[0,-1],[1,0],[2,0]])
            
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
            reconnection_distance = pde_solver.finger_width/2 + 2*pde_solver.ds
        else:
            reconnection_distance = 2*pde_solver.ds
            
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
        
        mask_outlet = np.logical_or(self.box.boundary_conditions==CONSTANT_FLUX,
                                    self.box.boundary_conditions==DIRICHLET_OUTLET)
        inds_outlet = np.where(mask_outlet)[0]
        # starting pt. ind.,
        # starting x, starting y, ending x, ending y
        pts_outlet = self.box.points[self.box.connections[inds_outlet]]
        all_segments_outlet = np.column_stack( ( inds_outlet,
                                                  pts_outlet[:,0],
                                                  pts_outlet[:,1])
                                                )
        
        
        for branch in self.active_branches:
            # BREAKTHROUGH
            min_distance, ind_min, is_pt_new, breakthrough_pt = \
                            find_reconnection_point(branch.points[-1], \
                                                all_segments_outlet[...,1:3], \
                                                all_segments_outlet[...,3:], 
                                                too_close=pde_solver.ds)
                                
            if min_distance < reconnection_distance:
                # decreasing step size while approaching BT:
                # remove False in if and uncomment pde_solver.ds=... in else below
                if False and pde_solver.ds>=1e-5:
                    pde_solver.ds = pde_solver.ds / 10
                    print("! Branch {ID} is reaching the outlet ! ds = {ds}".format(ID=branch.ID, ds=pde_solver.ds))
                else:
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
                    branch.steps = np.append(branch.steps, [step])
                    self.active_branches.remove(branch)
                    self.add_connection([branch.ID, -1])
                    
                    # contours based on the thickened tree
                    box_ring, _, _, _, _, _ = \
                        pde_solver.fingers_and_box_contours(self)
                    pde_solver.script_border_box, \
                        pde_solver.script_inside_buildmesh_box = \
                            pde_solver.prepare_script_box(self, box_ring)
            
            # RECONNECTION TO OTHER BRANCHES
            elif len(self.branches)>1 and branch.length() > 2*reconnection_distance:
                mask = np.ones(len(all_segments_branches), dtype=bool)        
                far_from_tip = sum(np.cumsum(np.flip(np.linalg.norm(branch.points[1:]-branch.points[:-1], axis=1)))>1.05*reconnection_distance)
                mask_branch = np.logical_and(all_segments_branches[:,0]==branch.ID, all_segments_branches[:,2]>=far_from_tip-1)
                mask[mask_branch] = False
                
                if type(pde_solver).__name__ == "FreeFEM_ThickFingers":
                    dr = branch.points[-1] - branch.points[-2]
                    dr = dr/np.linalg.norm(dr) * pde_solver.finger_width*0.6
                    tip = branch.points[-1] + dr
                else:
                    tip = branch.points[-1]
                    
                min_distance, ind_min, is_pt_new, reconnection_pt = \
                                find_reconnection_point(tip, \
                                                    all_segments_branches[mask,4:6], \
                                                    all_segments_branches[mask,6:], 
                                                    too_close=pde_solver.ds)                    

                if min_distance < reconnection_distance:
                    branch2_id = int(all_segments_branches[mask][ind_min,0])
                    print("! Branch {ID} reconnected to branch {ID2}!".format(ID=branch.ID, ID2=branch2_id))
                    
                    if type(pde_solver).__name__ == "FreeFEM_ThickFingers":
                        branch.points = np.vstack( (branch.points, [tip]) )
                        branch.steps = np.append(branch.steps, [step])
                    branch.points = np.vstack( (branch.points, [reconnection_pt]) )
                    branch.steps = np.append(branch.steps, [step])
                    self.active_branches.remove(branch)
                    self.add_connection([branch.ID, branch2_id])      
                    
                    if is_pt_new:
                        branch2_ind = int(all_segments_branches[mask][ind_min,1])
                        branch2 = self.active_branches[branch2_ind]
                        ind_pt = int(all_segments_branches[mask][ind_min, 2])
                        branch2.points = np.insert(branch2.points, ind_pt+1, reconnection_pt)
                        branch2.steps = np.insert(branch2.steps, ind_pt+1, branch2.steps[ind_pt])

                        print("New point... to test.")
                        # in theory we should update all_segments_branches
                        # all_segments_branches = all_segments_branches[ all_segments_branches[:,0]!=branch2_id]
                        # n_points = len(branch2.points)
                        # all_segments_branches = np.vstack(( all_segments_branches, \
                        #     np.column_stack( ( np.ones(n_points-1)*branch2.ID,
                        #                       np.ones(n_points-1)*branch2_ind,
                        #                       np.arange(n_points-1), np.arange(1, n_points),
                        #                       branch2.points[:-1], branch2.points[1:] )
                        #                     )
                        #     ) )
                

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

class System:
    """A class containing all the elements to run a network simulation.

    Attributes
    ----------
    network : Network
        An object of class Network.
    extender : Extender
        One of the classes from reticuler.extending_kernels.extenders.
    growth_thresh_type : int, default 0
        Type of growth threshold.
            - 0: max step
            - 1: height
            - 2: network length
    growth_thresh : float, default 5
        A value of growth threshold. The simulation is stopped, when it's reached.
    growth_gauges : array, default array([0.,0.,0.])
        A 1-3 array with growth gauges (max step, height, network length).
    dump_every : int, default 1
        Dumps the results every ``dump_every`` steps.
    exp_name: str, default ''
        Path to a file, where the results will be stored.

    """

    def __init__(
        self,
        network,
        extender,
        timestamps=None,
        growth_thresh_type=0,
        growth_thresh=5,
        growth_gauges=None,
        dump_every=1,
        exp_name="",
    ):
        """Initialize System.

        Parameters
        ----------
        network : Network
        extender : Extender from extending_kernels.extenders
        growth_gauges : array, default array([0.,0.,0.,0.])
        growth_thresh_type : int, default 0
        growth_thresh : float, default 5
        dump_every : int, default 1
        exp_name: str, default ''

        Returns
        -------
        None.

        """
        self.network = network
        self.extender = extender

        # Growth limits:
        # 0: max step, 1: max height
        # 2: max length 3: max time
        self.timestamps = np.array([0]) if timestamps is None else timestamps
        self.growth_gauges = np.zeros(4) if growth_gauges is None else growth_gauges
        self.growth_thresh_type = growth_thresh_type
        self.growth_thresh = growth_thresh

        self.dump_every = dump_every
        self.exp_name = exp_name
    
    def copy(self):
        """Return a deepcopy of the Network."""
        return copy.deepcopy(self)

    def export_json(self):
        """Export all the information to 'self.exp_name'+'.json'."""
        growth_type_legend = ["max step",
                              "max height", "max length", "max time"]
        export_general = {
            "reticuler_version": importlib.metadata.version("reticuler"),
            "exp_name": self.exp_name,
            "growth": {
                "dump_every": self.dump_every,
                "threshold_type": growth_type_legend[self.growth_thresh_type],
                "threshold": self.growth_thresh,
                "growth_gauges": {
                    "number_of_steps": self.growth_gauges[0],
                    "height": self.growth_gauges[1],
                    "network_length": self.growth_gauges[2],
                    "time": self.growth_gauges[3],
                },
            },
        }

        if type(self.extender).__name__ == "ModifiedEulerMethod":
            # if type(self.extender.pde_solver).__name__ == "FreeFEM":
            export_solver = {
                "type": type(self.extender.pde_solver).__name__,
                "description": "Equation legend: 0-Laplace, 1-Poisson. Bifurcation legend: 0-no bifurcations, 1-a1, 2-a3/a1, 3-random.",
                "is_script_saved": self.extender.pde_solver.is_script_saved,
                "equation": self.extender.pde_solver.equation,
                "eta": self.extender.pde_solver.eta,
                "ds": self.extender.pde_solver.ds,
                "bifurcation_type": self.extender.pde_solver.bifurcation_type,
                "bifurcation_thresh": self.extender.pde_solver.bifurcation_thresh,
                "bifurcation_angle": self.extender.pde_solver.bifurcation_angle,
                "inflow_thresh": self.extender.pde_solver.inflow_thresh,
                "distance_from_bif_thresh": self.extender.pde_solver.distance_from_bif_thresh,
            }
            if type(self.extender.pde_solver).__name__ == "FreeFEM_ThickFingers":
                export_solver["finger_width"] = self.extender.pde_solver.finger_width
                export_solver["mobility_ratio"] = self.extender.pde_solver.mobility_ratio
                
                
            export_extender = {
                "extender": {
                    "type": type(self.extender).__name__,
                    "max_approximation_step": self.extender.max_approximation_step,
                    "is_reconnecting": self.extender.is_reconnecting,
                    "pde_solver": {**export_solver},
                }
            }

        export_branches = {}
        for branch in self.network.branches[::-1]:
            if branch in self.network.active_branches:
                state = "active"
            elif branch in self.network.sleeping_branches:
                state = "sleeping"
            else:
                state = "dead"
            branch_dict = {
                branch.ID: {
                    "ID": branch.ID,
                    "state": state,
                    "points_and_steps": branch.points_steps(),
                }
            }
            export_branches = export_branches | branch_dict
        export_network = {
            "network": {
                "description": "Geometry of the system: box and branches.",
                "box": {
                    "description": "Border geometry. Points should be in a counterclokwise order. Connections and boundary conditions (BC) -> 1st/2nd columns: point IDs, 3rd column: BC. Seeds connectivity -> 1st column: index on border, 2nd column: branch ID.",
                    "points": self.network.box.points,
                    "connections_and_bc": self.network.box.connections_bc(),
                    "seeds_connectivity": self.network.box.seeds_connectivity,
                },
                "branch_connectivity": self.network.branch_connectivity,
                "branches": {**export_branches},
            }
        }

        export_timestamps = {"timestamps": self.timestamps}
        to_export = export_general | export_extender | \
                        export_network | export_timestamps
        with open(self.exp_name + ".json", "w", encoding="utf-8") as f:
            json.dump(to_export, f, ensure_ascii=False,
                      indent=4, cls=NumpyEncoder)

    @classmethod
    def import_json(cls, input_file):
        """Construct an instance of class System based on the imported .json file.

        Parameters
        ----------
        input_file : path
            Name of the experiment location. Extension '.json' will be added.

        Returns
        -------
        system : object of class System

        """
        with open(input_file + ".json", "r") as f:
            json_load = json.load(f)

        # Branches
        branches = []
        active_branches = []
        sleeping_branches = []
        for i in reversed(list(json_load["network"]["branches"].keys())):
            json_branch = json_load["network"]["branches"][i]
            points_steps = np.asarray(json_branch["points_and_steps"])
            branch = Branch(
                ID=json_branch["ID"],
                points=points_steps[:, :2],
                steps=np.array(points_steps[:, 2], dtype=int),
            )

            branches.append(branch)
            if json_branch["state"] == "active" or json_branch["state"] == "sleeping":
                active_branches.append(branch)
            # elif json_branch["state"] == "sleeping":
            #     sleeping_branches.append(branch)

        # Box
        json_box = json_load["network"]["box"]
        connections_bc = np.asarray(json_box["connections_and_bc"], dtype=int)
        box = Box(
            points=np.asarray(json_box["points"]),
            connections=connections_bc[:, :2],
            boundary_conditions=connections_bc[:, 2],
            seeds_connectivity=np.asarray(json_box["seeds_connectivity"], dtype=int),
        )
        # Network
        branch_connectivity = np.asarray(
            json_load["network"]["branch_connectivity"], dtype=int).reshape( \
                len(json_load["network"]["branch_connectivity"]), 2)
        network = Network(
            box=box,
            branches=branches,
            active_branches=active_branches,
            sleeping_branches=sleeping_branches,
            branch_connectivity=branch_connectivity,
        )
        # General
        json_growth = json_load["growth"]
        growth_type_legend = ["max step",
                              "max height", "max length", "max time"]
        growth_thresh_type = growth_type_legend.index(
            json_growth["threshold_type"])
        growth_thresh = json_growth["threshold"]
        dump_every = json_growth["dump_every"]
        timestamps = np.asarray(json_load["timestamps"])

        json_growth_gauges = json_growth["growth_gauges"]
        growth_gauges = np.array(
            [
                json_growth_gauges["number_of_steps"],
                json_growth_gauges["height"],
                json_growth_gauges["network_length"],
                json_growth_gauges["time"],
            ]
        )

        try:
            # Extender and solver
            json_extender = json_load["extender"]
            if json_extender["type"] == "ModifiedEulerMethod":  
                # Solver
                json_solver = json_load["extender"]["pde_solver"]
                if json_solver["type"] == "FreeFEM":
                    pde_solver_class = pde_solvers.FreeFEM
                elif json_solver["type"] == "FreeFEM_ThickFingers":
                    pde_solver_class = pde_solvers.FreeFEM_ThickFingers
                    
                json_solver.pop("type")
                json_solver.pop("description")
                pde_solver = pde_solver_class(network, **json_solver)
                # Extender
                extender = extenders.ModifiedEulerMethod(
                    pde_solver=pde_solver,
                    is_reconnecting=json_extender["is_reconnecting"],
                    max_approximation_step=json_extender["max_approximation_step"],
                )          
    
            system = cls(
                network=network,
                extender=extender,
                timestamps=timestamps,
                growth_gauges=growth_gauges,
                growth_thresh_type=growth_thresh_type,
                growth_thresh=growth_thresh,
                dump_every=dump_every,
                exp_name=input_file,
            )
        except Exception as error:
            print(type(error).__name__, ": ", error)
            print("!WARNING! Extender/PDE solver not imported.")
            pde_solver = pde_solvers.FreeFEM(network)
            extender = extenders.ModifiedEulerMethod(
                pde_solver=pde_solver,)
            system = cls(
                network=network,
                extender=extender,
                timestamps=timestamps,
                growth_gauges=growth_gauges,
                growth_thresh_type=growth_thresh_type,
                growth_thresh=growth_thresh,
                dump_every=dump_every,
                exp_name=input_file,
            )

        return system

    def __update_growth_gauges(self, dt):
        """Update growth gauges."""
        self.growth_gauges[1], self.growth_gauges[2] = self.network.height_and_length()
        self.growth_gauges[3] = self.growth_gauges[3] + dt
        self.timestamps = np.append( self.timestamps, self.growth_gauges[3] )

        print("Active branches: {n:d}".format(
            n=len(self.network.active_branches)))
        print("Network height: {h:.3f}".format(h=self.growth_gauges[1]))
        print("Network length: {l:.3f}".format(l=self.growth_gauges[2]))
        print("Evolution time: {t:.3f}".format(t=self.growth_gauges[3]))

    def evolve(self):
        """Run the simulation.

        Run the simulation in a while loop until ``self.growth_thresh`` is not reached.
        
        Returns
        -------
        None.

        """
        self.export_json()
        start_clock = time.time()
        while self.growth_gauges[self.growth_thresh_type] < self.growth_thresh \
                and len(self.network.active_branches) > 0:
            self.growth_gauges[0] = self.growth_gauges[0] + 1
            print(
                "\n-------------------   Growth step: {step:.0f}   -------------------\n".format(
                    step=self.growth_gauges[0]
                )
            )
            print("Date and time: ", datetime.datetime.now())

            # network evolution
            out_growth = self.extender.integrate(network=self.network, \
                                                 step=self.growth_gauges[0])
            
            self.__update_growth_gauges(out_growth[0])
            print("Computation time: {clock:.2f}h".format(
                    clock=(time.time() - start_clock)/3600
                )
            )
            
            if not self.growth_gauges[0] % self.dump_every:
                self.export_json()

        self.export_json()
        print("\n End of the simulation")

if __name__ == "__main__":
    box, branches, active_branches = Box.construct(
        initial_condition=4
    )