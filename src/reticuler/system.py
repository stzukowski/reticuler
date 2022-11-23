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
import pkg_resources

from .extending_kernels import extenders, pde_solvers, trajectory_integrators

class _NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.
    
    References
    ----------
    .. [1] https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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
    dR : array or 0, default 0
        A 1-2 array of tip progression ([dx, dy]).
    isBifurcating : bool, default False
        A boolean condition if branch is bifurcating or not.
        (Based on the bifurcation_type and bifurcation_thresh from extender.)
    isMoving : bool, default True
        A boolean condition if branch is moving further or not
        (Based on the inflow_thresh from extender.)
          
    """
    
    def __init__(self, ID, points, steps):
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
        
        self.points = points # in order of creation
        self.steps = steps # at which step of the evolution the point was added

        self.dR = 0
        self.isBifurcating = False
        self.isMoving = True
        
    def extend_by_dR(self):
        """Add a new point to self.points (progressed tip)."""
        self.points = np.vstack( (self.points, self.points[-1] + self.dR) )
                
    def length(self):
        """Return length of the Branch."""
        return np.sum( np.linalg.norm( (np.roll(self.points, -1, axis=0) - self.points), axis=1)[:-1])
    
    def tip_angle(self):
        """Return angle between the tip segment (last and penultimate point) and Y axis."""
        point_penult = self.points[-2]
        point_last = self.points[-1]
        dx = point_last[0] - point_penult[0]
        dy = point_last[1] - point_penult[1]
        return np.pi/2 - np.arctan2(dy, dx)
    
    def points_steps(self):
        """Return a 3-n array of points and evolution steps when they were added."""
        return np.column_stack( (self.points, self.steps) )


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
    
    def __init__(self, points=[], connections=[], boundary_conditions=[], seeds_connectivity=[]):
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
        self.points = points
        self.connections = connections
        self.boundary_conditions = boundary_conditions
        
        # 1st column: index on border
        # 2nd column: branch_id
        self.seeds_connectivity = seeds_connectivity
    
    def __add_points(self, points):
        if not len(self.points):
            self.points = points
        else:
            self.points = np.vstack( (self.points, points) )
                                
    def __add_connection(self, connections, boundary_conditions):
        if not len(self.connections):
            self.connections = connections
        else:
            self.connections = np.vstack( (self.connections,connections) )  
                              
        if not len(self.boundary_conditions):
            self.boundary_conditions = boundary_conditions
        else: 
            self.boundary_conditions = np.concatenates( (self.boundary_conditions,
                                                         boundary_conditions) )
                                                   
    def connections_bc(self):
        """Return a 3-n array of connections and boundary conditions corresponding to them.
        (1st/2nd column - point indices, 3rd column - BC)
        """
        return np.column_stack( (self.connections, self.boundary_conditions) )
    
    @classmethod
    def construct(cls, initial_condition=0, **kwargs_construct):
        """Construct a Box with given initial condition.
        
        Parameters
        ----------
        initial_condition : int, default 0
            IC = 0 or 1. Rectangular box of dimensions ``width`` x ``height``, \
            absorbing bottom wall, reflecting left and right, and:
                - IC = 0: constant flux on top (Laplacian case)
                - IC = 1: reflective top (Poissonian case)
        kwargs_construct:
            IC = 0 or 1
                seeds_x : array, default [0.1]
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
        box : object of class Box
        branches : list
            A list of objects of class Branch.
            

        """
        # Labels for boundary conditions
        DIRICHLET = 1
        NEUMANN = 2
        CONSTANT_FLUX = 3
        
        # Build a box
        box = cls()
        
        # Rectangular box of specified width and height
        # IC==0: Constant flux on top (Laplace)
        # IC==1: Neumann on top (Poisson)
        if initial_condition==0 or initial_condition==1:
            options_construct = {'seeds_x':[0.1], 'initial_lengths':[0.01], 'height':50.0, 'width':2.0}
            options_construct.update(kwargs_construct)
            box.__add_points( [ [options_construct['width'], 0], 
                                [options_construct['width'], options_construct['height']],
                                     [0, options_construct['height']], [0, 0] 
                              ])
            box.seeds_connectivity = np.column_stack( (
                                len(box.points) + np.arange(len(options_construct['seeds_x'])),
                                np.arange(len(options_construct['seeds_x'])) ) )
            box.__add_points( np.vstack( [options_construct['seeds_x'], 
                                          np.zeros(len(options_construct['seeds_x']))] ).T )
            
            connections_to_add=np.vstack( [ np.arange(len(box.points)), 
                                              np.roll(np.arange(len(box.points)), -1) ] ).T
            box.__add_connection( connections_to_add, 
                                  boundary_conditions=DIRICHLET * np.ones(len(connections_to_add), dtype=int)
                                 )          
              
            # right, left, top Neumann:
            box.boundary_conditions[0:3] = NEUMANN
            # or top constant flux:
            if initial_condition==0:
                box.boundary_conditions[1] = CONSTANT_FLUX
        
            # Creating initial branches
            branches = []
            if len(options_construct['initial_lengths'])==1:
                options_construct['initial_lengths'] = \
                                np.ones(len(options_construct['seeds_x']))*\
                                        options_construct['initial_lengths'][0]
            for i, x in enumerate(options_construct['seeds_x']):
                branches.append( Branch( ID=i, points=\
                                        np.array([ [x, 0], \
                                                  [x, options_construct['initial_lengths'][i]] ]), \
                                        steps=np.array([0, 0])) )
        
        return box, branches
    

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
    
    def __init__(self, box, branches=[], active_branches=[], sleeping_branches=[], branch_connectivity=[]):
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
        
        self.branches = branches # all branches (to construct mesh): moving + sleeping + branches inside the tree
        self.active_branches = active_branches # moving branches (to extend)
        self.sleeping_branches = sleeping_branches # branches without enough flux to move (may revive in the Poisson case)
        
        self.branch_connectivity = branch_connectivity
    
    def copy(self):
        """Return a deepcopy of the Network."""
        return copy.deepcopy(self)
    
    def height_and_length(self):
        """Return network height (max y coordinate) and total length of the branches."""
        ruler = 0
        height = 0
        for branch in self.branches:
            ruler = ruler + branch.length()
            height = np.max( (height, np.max(branch.points[:,1])) )
        return height, ruler
    
    def __add_connection(self, connection):
        """Add connection to self.branch_connectivity."""
        if not len(self.branch_connectivity):
            self.branch_connectivity = connection
        else:
            self.branch_connectivity = np.vstack( (self.branch_connectivity, connection) )
            
    def move_test_tips(self, dRs):
        """Move test tips (no bifurcations or killing).
        
        Assign ``dRs`` to self.dR and extend branches.
        
        """
        for i, branch in enumerate(self.active_branches):
            branch.dR = dRs[i]
            branch.extend_by_dR()
            
    def move_tips(self, step):
        """Move tips (with bifurcations and killing).     

        Parameters
        ----------
        step : int, default 0
            The current evolution step.

        Returns
        -------
        None.

        """
        
        # shallow copy of active_branches (creates new list instance, but the elements are still the same)
        branches_to_iterate = self.active_branches.copy()
        for i, branch in enumerate(branches_to_iterate):
            if not branch.isMoving:
                self.sleeping_branches.append(branch)
                self.active_branches.remove(branch)
                print('! Branch {ID} is sleeping !'.format(ID=branch.ID))
            else:
                if branch.isBifurcating:
                    print('! Branch {ID} bifurcated !'.format(ID=branch.ID))
                    max_branch_id = len(self.branches)-1
                    
                    for i, dR in enumerate(branch.dR):
                        points = np.array([ branch.points[-1], \
                                        branch.points[-1] + dR ])
                        branch_new = Branch( ID=max_branch_id+i+1, points=points, \
                                              steps=np.array([step-1]) ) 
                        self.branches.append(branch_new)
                        self.active_branches.append(branch_new)
                        self.__add_connection( np.array([branch.ID, branch_new.ID]) )
                    self.active_branches.remove(branch)
                else:
                    branch.extend_by_dR()


class System:
    """A class containing all the elements to run a network simulation.
    
    Attributes
    ----------
    network : Network
        An object of class Network.
    extender : Extender
        An object of one of the classes from reticuler.extending_kernels.extenders.
    trajectory_integrator : function
        One of the functions from reticuler.extending_kernels.trajectory_integrator. 
    growth_thresh_type : int, default 0
        Type of growth threshold.
            - 0: number of steps
            - 1: height
            - 2: network length
    growth_thresh : float, default 5
        A value of growth threshold. The simulation is stopped, when it's reached.
    growth_gauges : array, default array([0.,0.,0.])
        A 1-3 array with growth gauges (number of steps, height, network length).
    dump_every : int, default 10
        Dumps the results every ``dump_every`` steps.
    exp_name: str, default ''
        Path to a file, where the results will be stored.
      
    """
    
    def __init__(self, network, extender, trajectory_integrator,
                 growth_gauges=np.zeros(3), growth_thresh_type=0, growth_thresh=5,
                 dump_every=10, exp_name=''):
        """Initialize System.
        
        Parameters
        ----------
        network : Network
        extender : Extender
        trajectory_integrator : function
        growth_gauges : array, default array([0.,0.,0.])
        growth_thresh_type : int, default 0
        growth_thresh : float, default 5
        dump_every : int, default 10
        exp_name: str, default ''

        Returns
        -------
        None.

        """
        self.network = network
        self.extender = extender
        self.trajectory_integrator = trajectory_integrator
        
        # Growth limits:
        # 0: number of steps, 1: max height
        # 2: max tree length
        self.growth_gauges = growth_gauges
        self.growth_thresh_type = growth_thresh_type
        self.growth_thresh = growth_thresh
        
        self.dump_every = dump_every
        self.exp_name = exp_name        
        
    def export_json(self):
        """Export all the information to 'self.exp_name'+'.json'."""
        growth_type_legend = ['number of steps', 'max height', 'max tree length']
        export_general = {
            'reticuler_version': pkg_resources.get_distribution('reticuler').version,
            'exp_name': self.exp_name,
            'growth': {
                'threshold_type': growth_type_legend[self.growth_thresh_type],
                'threshold': self.growth_thresh,
                'growth_gauges': {
                    'number_of_steps': self.growth_gauges[0],
                    'height': self.growth_gauges[1],
                    'network_length': self.growth_gauges[2],
                    },
                'dump_every': self.dump_every
                }
            }
        
        if type(self.extender).__name__=='Streamline':
            export_trajectory_integrator = {
                'type': self.trajectory_integrator.__name__,
                }
            
            equation_legend = ['Laplace', 'Poisson']
            export_solver = {
                'type': type(self.extender.pde_solver).__name__,
                'equation': equation_legend[self.extender.pde_solver.equation],
                }
            
            bifurcation_type_legend = ['no bifurcations', 'a1', 'a3/a1', 'random']
            export_extender = {
                'extending_kernel': {
                    'trajectory_integrator': {
                        **export_trajectory_integrator
                        },
                    'extender': {
                        'type': type(self.extender).__name__,
                        'eta': self.extender.eta,
                        'ds': self.extender.ds,
                        'bifurcations': {
                            'type': bifurcation_type_legend[self.extender.bifurcation_type],
                            'threshold': self.extender.bifurcation_thresh,
                            'angle': self.extender.bifurcation_angle,
                            },
                        'inflow_thresh': self.extender.inflow_thresh,
                        'distance_from_bif_thresh': self.extender.distance_from_bif_thresh,
                        'pde_solver': {
                            **export_solver
                            }
                        }
                    }
                }
        
        export_branches = {}
        for branch in self.network.branches[::-1]:
            if branch in self.network.active_branches:
                state = 'active'
            elif branch in self.network.sleeping_branches:
                state = 'sleeping'
            else:
                state = 'dead'                
            branch_dict = { 
                branch.ID: {
                    'ID': branch.ID,
                    'state': state,
                    'points_and_steps': branch.points_steps(),
                    }
                }
            export_branches = export_branches | branch_dict
        export_network = {
            'network': {
                'description': 'Geometry of the system: box and branches.',
                'box': {
                    'description': 'Border geometry. Points should be in a counterclokwise order. Connections and boundary conditions (BC) -> 1st/2nd columns: point IDs, 3rd column: BC. Seeds connectivity -> 1st column: index on border, 2nd column: branch ID.',
                    'points': self.network.box.points,
                    'connections_and_bc': self.network.box.connections_bc(),
                    'seeds_connectivity': self.network.box.seeds_connectivity
                    },
                'branch_connectivity': self.network.branch_connectivity,
                'branches': {
                    **export_branches
                    }
                }
            }
        
        to_export = export_general | export_extender | export_network
        with open(self.exp_name+'.json', 'w', encoding='utf-8') as f:
            json.dump(to_export, f, ensure_ascii=False, indent=4, cls=_NumpyEncoder)
     
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
        with open(input_file+'.json', 'r') as f:
            json_load = json.load(f)
        
        # Branches
        branches = [] 
        active_branches = []
        sleeping_branches = []
        for i in reversed(list(json_load['network']['branches'].keys())):
            json_branch = json_load['network']['branches'][i]
            points_steps = np.asarray(json_branch['points_and_steps'])
            branch = Branch(ID=json_branch['ID'], points=points_steps[:,:2], steps=points_steps[:,2])
            
            branches.append(branch)
            if json_branch['state']=='active':
                active_branches.append(branch)
            elif json_branch['state']=='sleeping':
                sleeping_branches.append(branch)
                branch.isMoving = False
            elif json_branch['state']=='dead':
                branch.isMoving = False
            
        # Box
        json_box = json_load['network']['box']
        connections_bc = np.asarray(json_box['connections_and_bc'])
        box = Box( points=np.asarray(json_box['points']), connections=connections_bc[:,:2], \
                  boundary_conditions=connections_bc[:,2], seeds_connectivity=np.asarray(json_box['seeds_connectivity']) )
        # Network
        branch_connectivity = np.asarray(json_load['network']['branch_connectivity'])
        network = Network( box=box, branches=branches, active_branches=active_branches, \
                          sleeping_branches=sleeping_branches, branch_connectivity=branch_connectivity )
        
        # Trajectory integrator
        json_trajectory_integrator = json_load['extending_kernel']['trajectory_integrator']
        if json_trajectory_integrator['type']=='modified_euler':
            trajectory_integrator = trajectory_integrators.modified_euler
        
        # Solver
        json_solver = json_load['extending_kernel']['extender']['pde_solver']
        if json_solver['type']=='FreeFEM':
            equation_legend = ['Laplace', 'Poisson']
            equation = equation_legend.index(json_solver['equation'])          
            pde_solver = pde_solvers.FreeFEM(equation=equation)
        
        # Extender
        json_extender = json_load['extending_kernel']['extender']
        if json_extender['type']=='Streamline':
            json_bifurcation = json_extender['bifurcations']
            bifurcation_type_legend = ['no bifurcations', 'a1', 'a3/a1', 'random']
            bifurcation_type = bifurcation_type_legend.index(json_bifurcation['type'])
            extender = extenders.Streamline( pde_solver=pde_solver, \
                                            eta=json_extender['eta'], ds=json_extender['ds'], \
                                                bifurcation_type=bifurcation_type,\
                                                bifurcation_thresh=json_bifurcation['threshold'], \
                                                bifurcation_angle=json_bifurcation['angle'])
            extender.inflow_thresh = json_extender['inflow_thresh']
            extender.distance_from_bif_thresh = json_extender['distance_from_bif_thresh']
        
        # General
        json_growth = json_load['growth']
        growth_type_legend = ['number of steps', 'max height', 'max tree length']
        growth_thresh_type = growth_type_legend.index(json_growth['threshold_type'])
        growth_thresh = json_growth['threshold']
        dump_every = json_growth['dump_every']
        
        json_growth_gauges = json_growth['growth_gauges']
        growth_gauges = np.array([json_growth_gauges['number_of_steps'], json_growth_gauges['height'], json_growth_gauges['network_length']])
        
        system = cls( network=network, extender=extender,
                        trajectory_integrator=trajectory_integrator,
                        growth_gauges=growth_gauges,
                        growth_thresh_type=growth_thresh_type, growth_thresh=growth_thresh,
                        dump_every=dump_every, exp_name=json_load['exp_name'], )
        
        return system        
                
    def __update_growth_gauges(self):
        """Update growth gauges."""
        self.growth_gauges[1], self.growth_gauges[2] = self.network.height_and_length()
        # self.growth_gauges[3] = self.growth_gauges[3] + self.dt
        
        print('Active branches: {n:d}'.format(n=len(self.network.active_branches)))
        print('Network height: {h:.3f}'.format(h=self.growth_gauges[1]))
        print('Network length: {l:.3f}'.format(l=self.growth_gauges[2]))
        
        for branch in self.network.active_branches:
            branch.steps = np.append( branch.steps, self.growth_gauges[0] )
            # branch.timestamps = np.append( branch.timestamps, self.growth_gauges[3] )

                                        
    def evolve(self):
        """Run the simulation.
        
        Run the simulation in a while loop until ``self.growth_thresh`` is not reached.
        Print real time that the simulation took.

        Returns
        -------
        None.

        """
        start_clock = time.time()
        while self.growth_gauges[self.growth_thresh_type] < self.growth_thresh:
            self.growth_gauges[0] = self.growth_gauges[0] + 1
            print('\n-------------------   Growth step: {step:.0f}   -------------------\n'.format( step=self.growth_gauges[0]))
            print('Date and time: ', datetime.datetime.now())
            
            self.trajectory_integrator( extender=self.extender, network=self.network )
            self.network.move_tips( step=self.growth_gauges[0] )
            self.__update_growth_gauges()
            
            if not self.growth_gauges[0]%self.dump_every:
                self.export_json()
        
        self.export_json()
            
        print('\n End of the simulation. Time: {clock:.2f}s'.format(clock=time.time()-start_clock))