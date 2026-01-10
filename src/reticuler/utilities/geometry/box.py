"""Building blocks of the System.

Classes:
    Box
    Branch
    Network
"""

import numpy as np 
import copy

from reticuler.utilities.geometry.branch import Branch
from reticuler.utilities.misc import LEFT_WALL_PBC, RIGHT_WALL_PBC, DIRICHLET_1, DIRICHLET_0, NEUMANN_0, NEUMANN_1, DIRICHLET_0_GLOB_FLUX
from reticuler.utilities.misc import cyl2cart

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
            - 1: DIRICHLET_0, absorbing BC (vanishing field)
            - 2: DIRICHLET_1
            - 3: DIRICHLET_0_GLOB_FLUX, constant global flux
            - 4: NEUMANN_0, reflective BC (vanishing normal derivative)
            - 5: NEUMANN_1
            - 998: LEFT_WALL_PBC
            - 999: RIGHT_WALL_PBC
    seeds_connectivity : array, default []
        A 2-n array of seeds connectivity.
            - 1st column: index in ``points``
            - 2nd column: outgoing branch ``ID`` 

    """

    def __init__(
        self, points=None, connections=None, \
            boundary_conditions=None, seeds_connectivity=None, \
                initial_condition=None
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
        
        self.initial_condition = initial_condition
        
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
    def construct(cls, initial_condition=100, **kwargs_construct):
        """Construct a Box with given initial condition.

        Parameters
        ----------
        initial_condition : int, default 100
            IC = 100, 101, 102, 103, 300. Rectangular box of dimensions ``width`` x ``height``,
            absorbing bottom wall, reflecting left and right, and:
                - IC = 100: constant flux on top (Laplacian case)
                - IC = 101: reflective top (Poissonian case)
                - IC = 102: PBC right and left wall + DIRICHLET_1 BC on top
                - IC = 103: DIRICHLET_1 BC on top
                - IC = 300: DIRICHLET_1 BC on growing top
            IC = 200, 201: jellyfish (an octant) with a trifork
                - IC = 200: DIRICHLET_1 bottom and DIRICHLET_0_GLOB_FLUX on top
                            (u=1/0, but rescaled such that global flux is constant)
                - IC = 201: DIRICHLET_1 top and NEUMANN_1 bottom OR elasticity
                - IC = 202: DIRICHLET_1 bottom and DIRICHLET_0 bottom
            IC = 301: leaf semiellipse with seeds at the bottom boundary
            IC = 350: leaf circle with seeds in the center
            IC = 351: leaf slice with seeds in the center
        kwargs_construct:
            IC = 100, 101, 102, 103, 106, 300 (seeds vertically at the bottom)
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
            IC = 350, 351 (seeds radially in the center)
                seeds_phi : array, default [0]
                    A 1-n array of phi angles at the center (0,0) relative to Y axis.
                initial_lengths : array, default [0.4]
                    A 1-n array of seeds initial lengths.
                    Length must match seeds_x or be equal to 1 
                    (then the same initial length will be set for all seeds).
                radius : float, default 0.5
                    Radius of the semicircle/circle
                angular_width: float, default 2*np.pi
                    Angular width of the slice. If 2*np.pi, then initial_condition = 350.

        Returns
        -------
        box : Box
            An object of class Box.
        branches : list
            A list of objects of class Branch.


        """
        # Build a box
        box = cls(initial_condition=initial_condition)

        # Rectangular box of specified width and height
        if initial_condition//100==1 or initial_condition==300 or initial_condition==301:
            options_construct = {
                "seeds_x": [0.5],
                "initial_lengths": [0.01],
                "branch_BCs": [DIRICHLET_0],
                "height": 50,
                "width": 2,
            }
            options_construct.update(kwargs_construct)
            
            if type(options_construct["seeds_x"])==int:
                options_construct["seeds_x"]=options_construct["width"]/options_construct["seeds_x"]*(np.arange(options_construct["seeds_x"])+0.5)
            else:
                options_construct["seeds_x"]=np.array(options_construct["seeds_x"])
            
            if not len(options_construct["initial_lengths"])==len(options_construct["seeds_x"]):
                options_construct["initial_lengths"] = (
                    np.ones(len(options_construct["seeds_x"]))
                    * options_construct["initial_lengths"][0]
                )
            if not len(options_construct["branch_BCs"])==len(options_construct["seeds_x"]):
                options_construct["branch_BCs"] = (
                    np.ones(len(options_construct["seeds_x"]), dtype=int)
                    * options_construct["branch_BCs"][0]
                )
            options_construct["branch_BCs"]=np.array(options_construct["branch_BCs"])
            mask_seeds_from_outlet = options_construct["branch_BCs"]==DIRICHLET_1
            
            if initial_condition//100==1:
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
                            options_construct["height"]*np.ones(sum(mask_seeds_from_outlet)),
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
            else:
                n_points_top=int(50*options_construct["width"])
            
            if initial_condition==301:
                # semi ellipse
                box.__add_points(
                    np.vstack(( options_construct["width"]*np.cos(np.linspace(0, np.pi, n_points_top)),
                              options_construct["height"]*np.sin(np.linspace(0, np.pi, n_points_top)) )).T
                )
            if initial_condition==300:
                # bottom right corner
                box.__add_points([[options_construct["width"], 0]])
                # top including left and right corners
                box.__add_points(
                        np.vstack(
                            [
                                np.linspace(options_construct["width"], 0, n_points_top), #array of x
                                options_construct["height"]*np.ones(n_points_top), #array of y (constant)
                            ]
                        ).T
                    )
                #bottom left corner
                box.__add_points([[0, 0]])
                
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
            box.seeds_connectivity = np.vstack(
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
            box.boundary_conditions[:-1-sum(~mask_seeds_from_outlet)] = NEUMANN_0
            # or top constant flux:
            if initial_condition==100:
                box.boundary_conditions[1] = NEUMANN_1
            if initial_condition==102:
                box.boundary_conditions[1] = DIRICHLET_1
                box.boundary_conditions[0] = RIGHT_WALL_PBC
                box.boundary_conditions[2] = LEFT_WALL_PBC
            if initial_condition==103 or initial_condition==300:
                box.boundary_conditions[1:-2-sum(~mask_seeds_from_outlet)] = DIRICHLET_1
            if initial_condition==301:
                box.boundary_conditions[:-1-sum(~mask_seeds_from_outlet)] = DIRICHLET_1
                box.boundary_conditions[-1-sum(~mask_seeds_from_outlet):] = NEUMANN_0
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
        elif initial_condition//100==2:
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
            rim_pts_angs = np.linspace(-angular_width/2,angular_width/2, n_points_rim+1)[:-1]
            box.__add_points(
                cyl2cart(R_rim, rim_pts_angs, R_rim)
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
                boundary_conditions=DIRICHLET_1 # all walls DIRICHLET_1
                * np.ones(len(connections_to_add), dtype=int),
            )
            # right, left wall:
            box.boundary_conditions[0] = NEUMANN_0
            box.boundary_conditions[n_points_stomach+1] = NEUMANN_0
            if initial_condition==200:
                box.boundary_conditions[1:n_points_stomach+1] = DIRICHLET_0_GLOB_FLUX # top
            elif initial_condition==201:
                box.boundary_conditions[1:n_points_stomach+1] = DIRICHLET_0 # top
                box.boundary_conditions[n_points_stomach+2:] = NEUMANN_1 # bottom
            elif initial_condition==202:
                box.boundary_conditions[1:n_points_stomach+1] = DIRICHLET_0 # top
            else:
                raise ValueError(f"Initial condition {initial_condition} is incorrect! Choose another one.")

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
            # eps = np.array([0.013 , -0.012, 0.008, -0.011])
            eps = [0]*4
            # eps = np.random.uniform(low=-1, high=1, size=4)*0.2/R_rim
            for i, theta in enumerate(np.arange(-3/8,3.1/8,1/4)*angular_width):
                branch = Branch(
                        ID=3+i,
                        points=cyl2cart(np.array([R_rim, R_rim-0.075]), theta+eps[i], R_rim),
                        steps=np.array([0, 0])
                    )
                branches.append(branch)       
                active_branches.append(branch)
                # ind = box.seeds_connectivity[3+i, 0]
                ind = 2+n_points_stomach + np.argmin(np.abs(rim_pts_angs - (theta+eps[i])))
                box.points[ind] = branch.points[0]
                
            branch_connectivity = np.array([[0,-1],[1,0],[2,0]])
        
        # Leaf radial seeds in the center: circle or slice
        elif initial_condition//10==35:
            options_construct = {
                "seeds_phi": [0],
                "initial_lengths": [0.1],
                "branch_BCs": [DIRICHLET_0],
                "radius": 0.5,
                "angular_width": 2*np.pi,
            }
            options_construct.update(kwargs_construct)
            if options_construct["angular_width"]==2*np.pi:
                initial_condition = 350 # full circle
                box.initial_condition = 350 # full circle
            elif options_construct["angular_width"]<2*np.pi:
                initial_condition = 351 # slice
                box.initial_condition = 351 # slice

            if type(options_construct["seeds_phi"])==int:
                options_construct["seeds_phi"]=2*np.pi/options_construct["seeds_phi"]*np.arange(options_construct["seeds_phi"])
                
            if not len(options_construct["initial_lengths"])==len(options_construct["seeds_phi"]):
                options_construct["initial_lengths"] = (
                    np.ones(len(options_construct["seeds_phi"]))
                    * options_construct["initial_lengths"][0]
                )
            if not len(options_construct["branch_BCs"])==len(options_construct["seeds_phi"]):
                options_construct["branch_BCs"] = (
                    np.ones(len(options_construct["seeds_phi"]), dtype=int)
                    * options_construct["branch_BCs"][0]
                )
            options_construct["branch_BCs"]=np.array(options_construct["branch_BCs"])        
            
            n_points_rim = int(100*options_construct["radius"]*options_construct["angular_width"])
            
            # circular rim
            box.__add_points(
                np.vstack( cyl2cart(options_construct["radius"], \
                                    np.linspace(np.pi-options_construct["angular_width"]/2,\
                                                np.pi+options_construct["angular_width"]/2, \
                                                    n_points_rim), \
                                    0) )
            )
            if initial_condition==350:
                box.points=box.points[1:]
                box.seeds_connectivity = []
            else:
                box.__add_points([0,0])
                n_seeds = len(options_construct["seeds_phi"])
                box.seeds_connectivity = np.column_stack(
                        (
                            len(box.points)*np.ones(n_seeds),
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
            if initial_condition==351:
                box.boundary_conditions[-2:] = NEUMANN_0

            # Creating initial branches
            branches = []
            active_branches = []
            for i, phi in enumerate(options_construct["seeds_phi"]):
                BC = options_construct["branch_BCs"][i]
                IL = options_construct["initial_lengths"][i]
                branch = Branch(
                        ID=i,
                        points=np.array(
                            [[0, 0], [-IL*np.sin(phi), IL*np.cos(phi)]]
                        ),
                        steps=np.array([0, 0]),
                        BC=BC
                    )
                branches.append(branch)
                active_branches.append(branch)
            
            branch_connectivity = None
        else:
            raise ValueError(f"Initial condition {initial_condition} is incorrect! Choose another one.")
        
        return box, branches, active_branches, branch_connectivity
