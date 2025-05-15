"""Handle the Backward Evolution Algorithm.

Classes:
    Metrics
    Network
    System
    
"""

import time
import copy
import datetime
import numpy as np
import importlib.metadata
import json

from reticuler.system import NumpyEncoder
from reticuler.backward_evolution import trimmers


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
    nums_left : array
        A 1-n array with the number of points left in the forward branch at the current BEA step.
        Used to reconstruct the state of the network during BEA.
    steps : array
        A 1-n array with BEA steps at which corresponding points were added.
    flux_info : array
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

        self.points = np.empty((0,2), dtype=float)  # in order of trimming
        self.nums_left = [] # how many points are left in the forward branch
        self.steps = [] # at which step of the BEA the point was reached

        # arrays with BEA metrics are two elements shorter than points/steps
        # we don't collect metrics at the first and the last point of the initial branch
        self.flux_info = np.empty((0,3), dtype=float)
        self.overshoot = []
        self.angular_deflection = []
        
    def add_point(self, point, ind, BEA_step):
        """Add a new point to ``self.points`` (position of the tip after trimming)."""
        self.points = np.vstack((self.points, point))
        self.nums_left = np.append(self.nums_left, ind)
        self.steps = np.append(self.steps, BEA_step)

    def points_numsleft_steps(self):
        """Return a 3-n array of points, number of points left in the forward branch and BEA steps when they were added."""
        return np.column_stack((self.points, self.nums_left, self.steps))


class BackwardBifurcations:
    """A class containing BEA metrics at all bifurcation points.

    Attributes
    ----------
    mother_IDs : array
        A 1-n array with all IDs of the mother branchs of the bifurcations.
    flux_info : array
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
        
        self.flux_info = np.zeros((len(mother_IDs), 3))
        self.length_mismatch = np.zeros((len(mother_IDs), 1))
        
        # flag: 0-not reached, 1-reached once, 2-reached twice, 3-data saved
        self.flags = np.zeros(len(mother_IDs), dtype=int)
    
    def bif_info(self):
        """Return a 6-n array with bifurcation info (mother_ID, a1, a2, a3, length mismatch, flag)."""
        return np.column_stack((self.mother_IDs, self.flux_info, self.length_mismatch, self.flags))

class BackwardSystem:
    """A class containing all the elements to run the Backward Evolution Algorithm.

    Attributes
    ----------
    system : System
        An object of class System.
    trimmers : Trimmer
        An object of one of the classes from reticuler.BEA.trimmers.
    backward_branches : list
        A list of all backward branches.
    backward_bifurcations : BackwardBifurcations
        An object of class BackwardBifurcations.           
    BEA_step : int
        Current step of the BEA.
    BEA_step_thresh : int, default 1e4
        A value of BEA step threshold. The simulation is stopped, when it's reached.
    back_forth_steps_thresh : int, default 1
        Number of backward (and later forward) steps in one BEA step.
    dump_every : int, default 1
        Dumps the results every ``dump_every`` steps.
    exp_name: str, default  ``system.exp_name``+'_back'
        Path to a file, where the results will be stored.

    """

    def __init__(
        self,
        system,
        trimmer,
        BEA_step_thresh=1e4,
        back_forth_steps_thresh=1,
        dump_every=1,
    ):
        """Initialize System.

        Parameters
        ----------
        system : System
        trimmers : Trimmer
        back_forth_steps_thresh : int, default 1
        BEA_step_thresh : int, default 1e4
        dump_every : int, default 1

        Returns
        -------
        None.

        """
        self.system = system
        self.trimmer = trimmer

        self.back_forth_steps_thresh = back_forth_steps_thresh
        self.BEA_step = 0
        self.BEA_step_thresh = BEA_step_thresh

        # forward/backward branch ID is its index in backward_branches
        self.backward_branches, self.backward_bifurcations = self.import_branches(self.system.network)
        self.system.extender.pde_solver.eta = self.trimmer.eta
        self.system.extender.pde_solver.bifurcation_type = 0
        self.system.extender.pde_solver.inflow_thresh = 0
        self.system.extender.pde_solver.is_backward = True
        
        self.dump_every = dump_every
        self.exp_name = system.exp_name+'_back'
        
    @classmethod
    def import_json(cls, input_file, system):
        """Construct an instance of class BackwardSystem based on the imported .json file.

        Parameters
        ----------
        input_file : path
            Name of the experiment location. Extension '.json' will be added.

        Returns
        -------
        backward_system : object of class BackwardSystem

        """
        with open(input_file + ".json", "r") as f:
            json_load = json.load(f)

        # Backward branches
        backward_branches = []
        for i in reversed(list(json_load["backward_branches"].keys())):
            json_branch = json_load["backward_branches"][i]
            backward_branch = BackwardBranch(
                ID=json_branch["ID"],
                mother_ID=json_branch["mother_ID"],
            )
            
            points_numsleft_steps = np.asarray(json_branch["points_numsleft_steps"])
            if points_numsleft_steps.size:
                backward_branch.points = points_numsleft_steps[:, :2]
                backward_branch.nums_left = np.array(points_numsleft_steps[:, 2], dtype=int)
                backward_branch.steps = np.array(points_numsleft_steps[:, 3], dtype=int)
                backward_branch.flux_info = np.asarray(json_branch["flux_info"])
                backward_branch.overshoot = np.asarray(json_branch["overshoot"])
                backward_branch.angular_deflection = np.asarray(json_branch["angular_deflection"])

            backward_branches.append(backward_branch)

        # BackwardBifurcations
        bif_info = np.asarray(json_load["backward_bifurcations"]["bif_info"])
        backward_bifurcations = BackwardBifurcations(np.array(bif_info[:,0],dtype=int))
        backward_bifurcations.flux_info = bif_info[:, 1:4]
        backward_bifurcations.length_mismatch = bif_info[:, 4]
        backward_bifurcations.flags = np.array(bif_info[:,5], dtype=int)

        # Trimmer
        json_trimmer = json_load["trimmer"]
        if json_trimmer["type"] == "BackwardModifiedEulerMethod":
            trimmer = trimmers.BackwardModifiedEulerMethod(
                pde_solver=system.extender.pde_solver,
                eta=json_trimmer["eta"],
                ds=json_trimmer["ds"],
                max_approximation_step=json_trimmer["max_approximation_step"],
                inflow_thresh=json_trimmer["inflow_thresh"],
            )

        # General
        json_BEA_parameters = json_load["BEA_parameters"]
        BEA_step = json_BEA_parameters["BEA_step"]
        BEA_step_thresh = json_BEA_parameters["BEA_step_thresh"]
        back_forth_steps_thresh = json_BEA_parameters["back_forth_steps_thresh"]
        dump_every = json_BEA_parameters["dump_every"]

        backward_system = cls(
            system=system,
            trimmer=trimmer,
            BEA_step_thresh=BEA_step_thresh,
            back_forth_steps_thresh=back_forth_steps_thresh,
            dump_every=dump_every,
        )
        backward_system.BEA_step = BEA_step
        backward_system.backward_branches = backward_branches
        backward_system.backward_bifurcations = backward_bifurcations

        return backward_system

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
                "type": type(self.trimmer.pde_solver).__name__,
                "equation": equation_legend[self.trimmer.pde_solver.equation],
            }
        
        if type(self.trimmer).__name__ == "BackwardModifiedEulerMethod":
            export_trimmer = {
                    "trimmer": {
                        "type": type(self.trimmer).__name__,
                        "eta": self.trimmer.eta,
                        "ds": self.trimmer.ds,
                        "max_approximation_step": self.trimmer.max_approximation_step,
                        "inflow_thresh": self.trimmer.inflow_thresh,
                        "pde_solver": {**export_solver},
                    }
                }

        export_backward_bifurcations = {
            "backward_bifurcations": {
                "description": "Information gathered in the bifurcation points: mother_ID, a1, a2, a3, length mismatch, flag",
                "bif_info": self.backward_bifurcations.bif_info(),
            }
        }
        
        export_backward_branches = {}
        for branch in self.backward_branches[::-1]:
            branch_dict = {
                branch.ID: {
                    "ID": branch.ID,
                    "mother_ID": branch.mother_ID,
                    "points_numsleft_steps": branch.points_numsleft_steps(),
                    "flux_info": branch.flux_info,
                    "overshoot": branch.overshoot,
                    "angular_deflection": branch.angular_deflection,
                }
            }
            export_backward_branches = export_backward_branches | branch_dict
        export_backward_branches = { "backward_branches": {**export_backward_branches} }
        
        to_export = export_general | export_trimmer | export_backward_bifurcations | export_backward_branches
        with open(self.exp_name + ".json", "w", encoding="utf-8") as f:
            json.dump(to_export, f, ensure_ascii=False,
                      indent=4, cls=NumpyEncoder)

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
            
            if network.branch_connectivity.size==0:
                mother_ID = -1
                network.active_branches.append(forward_branch)
            else:
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
            if network.branch_connectivity.size!=0 and \
                np.isin(forward_branch.ID, network.branch_connectivity[:,0], invert=True):
                network.active_branches.append(forward_branch)

        # above, we go over branches, so the mother_IDs are repeated; hence np.unique
        backward_bifurcations = BackwardBifurcations(np.unique(mother_IDs))

        return backward_branches, backward_bifurcations
    
    def copy(self):
        """Return a deepcopy of the BacwkardSystem."""
        return copy.deepcopy(self)

    def __compare_networks(self, initial_network, backward_network, test_network, flux_info):
        """Gathering BEA metrics.
        
        Parameters
        ----------
        initial_network : Network
        backward_network : Network
            Network after backward step.
        test_network : Network
            Network after backward-forward step.
        flux_info : array
            A 3-n array with a1,a2,a3 coefficients after backward step.

        Returns
        -------
        None.
        
        """

        for i in range(len(initial_network.active_branches)):
            backward_branch = self.backward_branches[initial_network.active_branches[i].ID]
            bifurcation_ind = self.backward_bifurcations.mother_IDs == \
                backward_branch.mother_ID
                
            # if we've reached the bifurcation point for the second time
            if self.backward_bifurcations.flags[bifurcation_ind]==2:
                self.backward_bifurcations.flags[bifurcation_ind] = 3
                self.backward_bifurcations.flux_info[bifurcation_ind] = flux_info[i]
                # length_mismatch is saved in the trimmer after first visit in the bifurcation point
        
            # if we haven't reached the bifurcation point
            # (if we've reached it only once then the branched is popped from active_branches)
            else:
                initial_point = initial_network.active_branches[i].points[-1]
                back_point = backward_network.active_branches[i].points[-1]
                test_point = test_network.active_branches[i].points[-1]
    
                # for angular deflection
                v1 = initial_point - back_point
                real_angle = np.arctan2(v1[1], v1[0])
                v2 = test_point - back_point
                test_angle = np.arctan2(v2[1], v2[0])
    
                # overshoot
                backward_branch.overshoot = np.append(backward_branch.overshoot, \
                                                      np.linalg.norm(test_point - initial_point) / \
                                                          np.linalg.norm(initial_point - back_point))
                    
                # angular deflection
                backward_branch.angular_deflection = np.append(backward_branch.angular_deflection, \
                                                               real_angle-test_angle)
                # a1, a2, a3 coefficients
                backward_branch.flux_info = np.vstack((backward_branch.flux_info, flux_info[i]))
                
    def run_BEA(self):
        """Run the Backward Evolution Algorithm.

        Returns
        -------
        None.

        """
        start_clock = time.time()
        backward_dts = np.empty(self.back_forth_steps_thresh)
        # while branches list is not empty
        while not len(self.system.network.branches)==1 and self.BEA_step < self.BEA_step_thresh:
            self.BEA_step = self.BEA_step + 1
            print(
                "\n-------------------   Backward Evolution Algorithm step: {step:.0f}   -------------------\n".format(
                    step=self.BEA_step
                )
            )
            print("Date and time: ", datetime.datetime.now())

            ##### BACKWARD STEPS #####
            print("Backward steps")
            for i in range(self.back_forth_steps_thresh):
                initial_network, self.system.network, \
                self.backward_branches, self.backward_bifurcations, \
                backward_dts[i] = \
                    self.trimmer.trim(
                        self.system.network, self.backward_branches, \
                        self.backward_bifurcations, self.BEA_step)

            # if there are no living branches we skip forward steps
            if not self.system.network.active_branches:
                continue

            ##### FORWARD STEPS #####
            print("----- Forward steps -----")
            test_network = self.system.network.copy()
            for i in range(self.back_forth_steps_thresh):
                self.system.extender.pde_solver.ds = backward_dts[i]
                _, flux_info = self.system.extender.integrate(network=test_network, \
                                     step=self.system.growth_gauges[0], is_dr_normalized=False) # dt, flux_info


            # compare the network before and after the backward-forward steps
            self.__compare_networks(
                initial_network, self.system.network, test_network, flux_info)

            print("Computation time: {clock:.2f}h".format(
                    clock=(time.time() - start_clock)/3600
                )
            )
            if not self.BEA_step % self.dump_every:
                self.export_json()
                # self.system.export_json()

        self.export_json()
        # self.system.export_json()

        print("\n End of the Backward Evolution Algorithm.")        
