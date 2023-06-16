"""Handle the Backward Evolution Algorithm.

Classes:
    Metrics
    Network
    System
    
"""

import time
import datetime
import numpy as np
import importlib.metadata
import json

from reticuler.system import NumpyEncoder


class BackwardBranch:
    """A class of a single branch in a network used to collect BEA metrics.

    Attributes
    ----------
    ID : int
        Branch ID.
    mother_ID : int
        ID of the mother branch (branch from which `self` bifrucated). 
        If `self` is connected to the border mother_ID=-1.
    points : array
        A 2-n array with xy coordinates of the points composing the branch.
        Chronological order of growth (tip is the last point).
    steps : array
        A 1-n array with BEA steps at which corresponding points were added.
    a1a2a3_coefficients : array
        A 3-n array with a1,a2,a3 coefficients in a point after backward step.
    overshoot : array
        A 1-n array with overshoot after virtual forward step.
    angular_deflection : array
        A 1-n array with angular deviation after virtual forward step.

    """

    def __init__(self, ID, mother_ID):
        """Initialize BackwardBranch.

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
        self.mother_ID = mother_ID

        self.points = []  # in order of trimming
        self.steps = [] # at which step of the BEA the point was reached

        # arrays with BEA metrics are two elements shorter than points/steps
        # we don't collect metrics at the first and the last point of the initial branch
        self.a1a2a3_coefficients = []
        self.overshoot = []
        self.angular_deflection = []
        
    def add_point(self, point, BEA_step):
        """Add a new point to ``self.points`` (position of the tip after trimming)."""
        self.points = np.vstack((self.points, point))
        self.steps = np.append(self.steps, BEA_step)

    def points_steps(self):
        """Return a 3-n array of points and BEA steps when they were added."""
        return np.column_stack((self.points, self.steps))


class BackwardBifurcations:
    """A class containing BEA metrics at all bifurcation points.

    Attributes
    ----------
    mother_IDs : array
        A 1-n array with all IDs of the mother branchs of the bifurcations.
    a1a2a3_coefficients : array
        A 3-n array with a1,a2,a3 coefficients in each bifurcation point 
        after two branches reached it.
    length_mismatch : array
        A 1-n array with length mismatch in each bifurcation point
        after one of the branches reached it.
    flags : array
        A 1-n array of flags:
            - 0: bifurcation not reached
            - 1: reached once
            - 2: reached twice
            - 3: a1a2a3 coefficients stored

    """
    def __init__(self, mother_IDs):
        """Initialize BackwardBifurcations.

        Parameters
        ----------
        mother_IDs : array

        Returns
        -------
        None.

        """
        self.mother_IDs = mother_IDs
        
        self.a1a2a3_coefficients = np.zeros((len(mother_IDs), 3))
        self.length_mismatch = np.zeros((len(mother_IDs), 1))
        
        # flag: 0-not reached, 1-reached once, 2-reached twice, 3-data saved
        self.flags = np.zeros((len(mother_IDs), 1))
    
    def bif_info(self):
        """Return a 6-n array with bifurcation info (mother_ID, a1, a2, a3, length mismatch, flag)."""
        return np.column_stack((self.mother_IDs, self.a1a2a3_coefficients, self.length_mismatch, self.flags))


class BackwardEvolutionAlgorithm:
    """A class containing all the elements to run the Backward Evolution Algorithm.

    Attributes
    ----------
    system : System
        An object of class System.
    trimmers : Trimmer
        An object of one of the classes from reticuler.backward_evolution_algorithm.trimmers.
    backward_branches : list
        A list of all backward branches.
    backward_bifurcations : BackwardBifurcations
        An object of class BackwardBifurcations.           
    BEA_step : int
        Current step of the BEA.
    BEA_step_thresh : int, default 1e4
        A value of BEA step threshold. The simulation is stopped, when it's reached.
    back_forth_steps_thresh : int, default 1
        Numeber of backward (and later forward) steps in one BEA step.
    dump_every : int, default 1
        Dumps the results every ``dump_every`` steps.
    exp_name: str, default ''
        Path to a file, where the results will be stored.     

    """

    def __init__(
        self,
        system,
        trimmer,
        BEA_step_thresh=1e4,
        back_forth_steps_thresh=1,
        dump_every=1,
        exp_name='',
    ):
        """Initialize System.

        Parameters
        ----------
        system : System
        trimmers : Trimmer
        back_forth_steps_thresh : int, default 1
        BEA_step_thresh : int, default 1e4
        dump_every : int, default 1
        exp_name: str, default ''

        Returns
        -------
        None.

        """
        self.system = system
        self.trimmer = trimmer

        self.back_forth_steps_thresh = back_forth_steps_thresh
        self.BEA_step = 1
        self.BEA_step_thresh = BEA_step_thresh

        # forward/backward branch ID is its index in backward_branches
        self.backward_branches, self.backward_bifurcations = self.import_branches(self.system.network)
        self.system.extender.eta = self.trimmer.eta
        self.system.extender.bifurcation_type = 0
        self.system.extender.inflow_thresh = 0
        
        self.dump_every = dump_every
        self.exp_name = exp_name

    def import_branches(self, network):
        """Reorganizing and importing branches from the networks.
        
        Parameters
        ----------
        network : Network

        Returns
        -------
        backward_branches : list
        backward_bifurcations : BackwardBifurcations
        
        """
        network.active_branches = []
        network.sleeping_branches = []
        
        branch_connectivity_0 = network.branch_connectivity.copy()
        seeds_connectivity_0 = network.box.seeds_connectivity.copy()
        
        backward_branches = []
        mother_IDs = []
        for i, forward_branch in enumerate(network.branches):
            network.branch_connectivity[branch_connectivity_0 ==
                                        forward_branch.ID] = i
            network.box.seeds_connectivity[seeds_connectivity_0[:, 1]
                                           == forward_branch.ID, 1] = i
            forward_branch.ID = i
            
            mother = network.branch_connectivity[
                network.branch_connectivity[:,1]==i, 0]
            if mother.size > 0:
                mother_ID = mother[0]
                mother_IDs.append(mother[0])
            else:
                mother_ID = -1
                
            # forward/backward branch ID is its index in backward_branches
            backward_branches.append(BackwardBranch(ID=i, mother_ID=mother_ID))
            
            # free node ==> active branch
            if np.isin(forward_branch.ID, network.branch_connectivity[:,0], invert=True):
                network.active_branches.append(forward_branch)

        # above, we go over branches, so the mother_IDs are repeated; hence np.unique
        backward_bifurcations = BackwardBifurcations(np.unique(mother_IDs))

        return backward_branches, backward_bifurcations

    def run(self):
        """Run the Backward Evolution Algorithm.

        Returns
        -------
        None.

        """
        start_clock = time.time()
        a1a2a3_coefficients = np.empty((self.back_forth_steps_thresh, 3))
        backward_dts = np.empty(self.back_forth_steps_thresh)
        # while branches list is not empty
        while not len(self.system.network.branches)==1 or self.BEA_step < self.BEA_step_thresh:
            self.BEA_step = self.BEA_step + 1
            print(
                "\n-------------------   Backward Evolution Algorith step: {step:.0f}   -------------------\n".format(
                    step=self.BEA_step
                )
            )
            print("Date and time: ", datetime.datetime.now())

            ##### BACKWARD STEPS #####
            for i in range(self.back_step_thresh):
                initial_network, self.system.network, backward_dts[i] = \
                    self.trimmer.trim(
                        self.system.network, self.backward_branches, self.backward_bifurcations, self.BEA_step)

            # if there are no living branches we skip forward steps
            if not self.system.network.active_branches:
                continue

            ##### FORWARD STEPS #####
            test_network = self.system.network.copy()
            for i in range(self.back_step_thresh):
                self.system.extender.ds = backward_dts[i]
                _, a1a2a3_coefficients[i] = \
                    self.system.trajectory_integrator.integrate(
                        extender=self.system.extender,
                        network=self.system.network,
                        is_BEA_off=False
                )
                test_network.move_tips(step=self.growth_gauges[0])

            # compare the network before and after the backward-forward steps
            self.__compare_networks(
                initial_network, self.system.network, test_network, a1a2a3_coefficients)

            if not self.BEA_step % self.dump_every:
                self.export_json()

        self.export_json()

        print(
            "\n End of the Backward Evolution Algorithm. Time: {clock:.2f}s".format(
                clock=time.time() - start_clock
            )
        )        

    def __compare_networks(self, initial_network, backward_network, test_network, a1a2a3_coefficients):
        """Gathering BEA metrics.
        
        Parameters
        ----------
        initial_network : Network
        backward_network : Network
            Network after backward step.
        test_network : Network
            Network after backward-forward step.
        a1a2a3_coefficients : array
            A 3-n array with a1,a2,a3 coefficients after backward step.

        Returns
        -------
        None.
        
        """

        for i in range(len(initial_network.active_branches)):
            backward_branch = self.backward_branches[initial_network.active_branches[i].ID]
            bifurcation_ind = self.backward_bifurcations.mother_IDs == \
                initial_network.active_branches[i].ID
                
            # if we've reached the bifurcation point for the second time
            if self.backward_bifurcations.flags[bifurcation_ind]==2:
                self.backward_bifurcations.flags[bifurcation_ind] = 3
                self.backward_bifurcations.a1a2a3_coefficients[bifurcation_ind] = a1a2a3_coefficients[bifurcation_ind]
                # length_mismatch is saved in the trimmer after first visit in the bifurcation point
        
            # if we haven't reached the bifurcation point
            # (if we've reached it only once then the branched is popped from active_branches)
            else:
                initial_point = initial_network.active_branches[i].points[-1]
                back_point = backward_network.active_branches[i].points[-1]
                test_point = test_network.active_branches[i].points[-1]
    
                # for angular deflection
                v1 = initial_point - back_point
                v1 = v1 / np.linalg.norm(v1)
                v2 = test_point - back_point
                v2 = v2 / np.linalg.norm(v2)
    
                # overshoot
                backward_branch.overshoot.append(
                    np.linalg.norm(test_point - initial_point))
                # angular deflection
                backward_branch.angular_deflection.append(
                    np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
                # a1, a2, a3 coefficients
                backward_branch.a1a2a3_coefficients.append(a1a2a3_coefficients[i])
                
    def export_json(self):
        """Export all the information to 'self.exp_name'+'.json'."""
        
        export_general = {
            "reticuler_version": importlib.metadata.version("reticuler"),
            "exp_name": self.exp_name,
            "BEA_parameters": {
                "back_forth_steps_thresh": self.back_forth_steps_thresh,
                "BEA_step": self.BEA_step,
                "BEA_step_thresh": self.BEA_step_thresh,
                "dump_every": self.dump_every,
            },
        }
        
        if type(self.trimmer.pde_solver).__name__ == "FreeFEM":
            equation_legend = ["Laplace", "Poisson"]
            export_solver = {
                "type": type(self.extender.pde_solver).__name__,
                "equation": equation_legend[self.extender.pde_solver.equation],
            }
        
        if type(self.trimmer).__name__ == "BackwardModifiedEulerMethod":
            export_trimmer = {
                "type": type(self.trimmer).__name__,
                "eta": self.extender.eta,
                "ds": self.extender.ds,
                "max_approximation_step": self.trajectory_integrator.max_approximation_step,
                "inflow_thresh": self.extender.inflow_thresh,
                "pde_solver": {**export_solver},
            }

        export_backward_bifurcations = {
            "description": "Information gathered in the bifurcation points: mother_ID, a1, a2, a3, length mismatch, flag",
            "bif_info": self.bif_info(),
        }

        export_backward_branches = {}
        for branch in self.backward_branches[::-1]:
            branch_dict = {
                branch.ID: {
                    "ID": branch.ID,
                    "mother_ID": branch.mother_ID,
                    "points_and_steps": branch.points_steps(),
                    "a1a2a3_coefficients": branch.a1a2a3_coefficients,
                    "overshoot": branch.overshoot,
                    "angular_deflection": branch.angular_deflection,
                }
            }
            export_backward_branches = export_backward_branches | branch_dict
    
        to_export = export_general | export_trimmer | export_backward_bifurcations | export_backward_branches
        with open(self.exp_name + ".json", "w", encoding="utf-8") as f:
            json.dump(to_export, f, ensure_ascii=False,
                      indent=4, cls=NumpyEncoder)

