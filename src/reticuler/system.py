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
import pickle
import importlib.metadata

import matplotlib.pyplot as plt

from reticuler import NumpyEncoder
from reticuler import Branch, Box, Network
from reticuler.extending_kernels import extenders
from reticuler.extending_kernels import pde_solvers
from reticuler.utilities import morphers

from reticuler.user_interface import graphics

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
    growth_gauges : array, default array([0.,0.,0.,0.])
        A 1-4 array with growth gauges (max step, height, network length, evolution time).
    dump_every : int, default 1
        Dumps the results every ``dump_every`` steps.
    exp_name: str, default ''
        Path to a file, where the results will be stored.

    """

    def __init__(
        self,
        network,
        extender,
        morpher=None,
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
        if extender.pde_solver and "FreeFEM" in type(extender.pde_solver).__name__:
            self.extender.pde_solver.system = self
        self.morpher = morpher

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
            export_solver = {
                "type": type(self.extender.pde_solver).__name__,
                "description": "",
                "is_script_saved": self.extender.pde_solver.is_script_saved,
                "equation": self.extender.pde_solver.equation,
                "eta": self.extender.pde_solver.eta,
                "ds": self.extender.pde_solver.ds,
            }
            if "FreeFEM_ThinFingers" in type(self.extender.pde_solver).__name__:
                export_solver["description"] = "Equation legend: 0-Laplace, 1-Poisson. Bifurcation legend: 0-no bifurcations, 1-a1, 2-a3/a1, 3-random."
                export_solver["bifurcation_type"] = self.extender.pde_solver.bifurcation_type
                export_solver["bifurcation_thresh"] = self.extender.pde_solver.bifurcation_thresh
                export_solver["bifurcation_angle"] = self.extender.pde_solver.bifurcation_angle
                export_solver["inflow_thresh"] = self.extender.pde_solver.inflow_thresh
                export_solver["distance_from_bif_thresh"] = self.extender.pde_solver.distance_from_bif_thresh
            elif type(self.extender.pde_solver).__name__ == "FreeFEM_ThickFingers":
                export_solver["description"] = "Equation legend: 0-Laplace, 1-Poisson."
                export_solver["finger_width"] = self.extender.pde_solver.finger_width
                export_solver["mobility_ratio"] = self.extender.pde_solver.mobility_ratio
            elif type(self.extender.pde_solver).__name__ == "FreeFEM_ThickFingers_Elasticity":
                export_solver["description"] = "Equation legend: 2-elasticity."
                export_solver["finger_width"] = self.extender.pde_solver.finger_width
                export_solver["youngs_modulus_inside"] = self.extender.pde_solver.youngs_modulus_inside
                export_solver["youngs_modulus_outside"] = self.extender.pde_solver.youngs_modulus_outside
                export_solver["poissons_ratio_inside"] = self.extender.pde_solver.poissons_ratio_inside
                export_solver["poissons_ratio_outside"] = self.extender.pde_solver.poissons_ratio_outside
                
            export_extender = {
                "extender": {
                    "type": type(self.extender).__name__,
                    "max_approximation_step": self.extender.max_approximation_step,
                    "is_reconnecting": self.extender.is_reconnecting,
                    "pde_solver": {**export_solver},
                }
            }

        if self.morpher is not None:
            if type(self.morpher).__name__ == "Jellyfish":
                export_morpher = {
                    "morpher": {
                        "type": type(self.morpher).__name__,
                        "timescale": self.morpher.timescale,
                        "v_rim": self.morpher.v_rim,
                        "sprouting_thresh": self.morpher.sprouting_thresh,
                        "sprouting_stochastic_factor": self.morpher.sprouting_stochastic_factor,
                        "radii": self.morpher.radii,
                        }
                    }
            elif type(self.morpher).__name__ == "Leaf":
                # export_box_history = {}
                # for i, bx in enumerate(self.morpher.box_history):
                #     box_dict = { f"box{i}": {
                #         "points": bx.points,
                #         "connections_and_bc": bx.connections_bc(),
                #         "seeds_connectivity": bx.seeds_connectivity,
                #         "initial_condition": bx.initial_condition,
                #         }
                #     }
                #     export_box_history = export_box_history | box_dict
                
                with open(self.exp_name + "_box_history.pkl", "wb") as f:
                    pickle.dump(self.morpher.box_history, f)

                export_morpher = {
                    "morpher": {
                        "type": type(self.morpher).__name__,
                        "v_rim": self.morpher.v_rim,
                        "box_history": "pickled" # export_box_history
                        }
                    }                
        else:
            export_morpher = {}
                
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
                    "BC": branch.BC,
                    "points_and_steps": branch.points_steps(),
                }
            }
            export_branches = export_branches | branch_dict
        export_network = {
            "network": {
                "description": "Geometry of the system: box and branches.",
                "box": {
                    "description": "Border geometry. Points should be in a counterclokwise order. Connections and boundary conditions (BC) -> 1st/2nd columns: point IDs, 3rd column: BC. Seeds connectivity -> 1st column: index on border, 2nd column: branch ID.",
                    "initial_condition": self.network.box.initial_condition,
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
                        export_morpher | export_network | export_timestamps
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
            # !!! Backward compatibility
            try:
                branch = Branch(
                    ID=json_branch["ID"],
                    points=points_steps[:, :2],
                    steps=np.array(points_steps[:, 2], dtype=int),
                    BC=json_branch["BC"],
                )
            except:
                branch = Branch(
                    ID=json_branch["ID"],
                    points=points_steps[:, :2],
                    steps=np.array(points_steps[:, 2], dtype=int),
                )

            branches.append(branch)
            if json_branch["state"] == "active" or json_branch["state"] == "sleeping":
                active_branches.append(branch)

        # Box
        json_box = json_load["network"]["box"]
        connections_bc = np.asarray(json_box["connections_and_bc"], dtype=int)
        # !!! Backward compatibility
        try:
            initial_condition = json_box["initial_condition"]
        except:
            initial_condition = None
        box = Box(
            points=np.asarray(json_box["points"]),
            connections=connections_bc[:, :2],
            boundary_conditions=connections_bc[:, 2],
            seeds_connectivity=np.asarray(json_box["seeds_connectivity"], dtype=int),
            initial_condition=initial_condition,
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
        
        try:
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

            # Morpher
            if "morpher" in json_load:
                json_morpher = json_load["morpher"]
                if json_morpher["type"] == "Jellyfish":
                    # !!! Backward compatibility
                    try:
                        sprouting_thresh=json_morpher["sprouting_thresh"]
                        sprouting_stochastic_factor=json_morpher["sprouting_stochastic_factor"]
                    except Exception as error:
                        sprouting_thresh = 1.5
                        sprouting_stochastic_factor = 0.2                    
                    morpher = morphers.Jellyfish(
                                    radii=json_morpher["radii"],
                                    timescale=json_morpher["timescale"],
                                    v_rim=json_morpher["v_rim"],
                                    sprouting_thresh=sprouting_thresh,
                                    sprouting_stochastic_factor=sprouting_stochastic_factor,
                                    )
                elif json_morpher["type"] == "Leaf":
                    # !!! Backward compatibility
                    try:
                        with open(input_file + "_box_history.pkl", "rb") as f:
                            boxes = pickle.load(f)
                    except Exception as error:
                        print(type(error).__name__, ": ", error)
                        print("Importing box history the old way.")
                        boxes = []
                        for box_ind in json_morpher["box_history"]:
                            json_box = json_morpher["box_history"][box_ind]
                            connections_bc = np.asarray(json_box["connections_and_bc"], dtype=int)
                            initial_condition = None
                            box = Box(
                                points=np.asarray(json_box["points"]),
                                connections=connections_bc[:, :2],
                                boundary_conditions=connections_bc[:, 2],
                                seeds_connectivity=np.asarray(json_box["seeds_connectivity"], dtype=int),
                                initial_condition=initial_condition,
                            )  
                            boxes.append(box.copy())
                    morpher = morphers.Leaf(
                                    v_rim=json_morpher["v_rim"],
                                    box_history=boxes,
                                    )

            else:
                morpher = None
            
            # Extender and solver
            json_extender = json_load["extender"]
            if json_extender["type"] == "ModifiedEulerMethod":  
                # Solver
                json_solver = json_load["extender"]["pde_solver"]
                # !!! Backward compatibility
                if morpher is None and (json_solver["type"] == "FreeFEM" or json_solver["type"] == "FreeFEM_ThinFingers"):
                    pde_solver_class = pde_solvers.FreeFEM_ThinFingers
                elif json_solver["type"] == "FreeFEM_ThinFingers_Boundary" or json_morpher["type"] == "Leaf":
                    pde_solver_class = pde_solvers.FreeFEM_ThinFingers_Boundary
                elif json_solver["type"] == "FreeFEM_ThickFingers":
                    pde_solver_class = pde_solvers.FreeFEM_ThickFingers
                    json_solver.pop("bifurcation_type", None)
                    json_solver.pop("bifurcation_thresh", None)
                    json_solver.pop("bifurcation_angle", None)
                    json_solver.pop("inflow_thresh", None)
                    json_solver.pop("distance_from_bif_thresh", None)
                elif json_solver["type"] == "FreeFEM_ThickFingers_Elasticity":
                    pde_solver_class = pde_solvers.FreeFEM_ThickFingers_Elasticity
                    
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
                morpher=morpher,
                timestamps=timestamps,
                growth_gauges=growth_gauges,
                growth_thresh_type=growth_thresh_type,
                growth_thresh=growth_thresh,
                dump_every=dump_every,
                exp_name=input_file,
            )
        except Exception as error:
            print(type(error).__name__, ": ", error)
            print("!WARNING! Extender/PDE solver/morpher not imported.")
            pde_solver = pde_solvers.FreeFEM_ThinFingers(network)
            extender = extenders.ModifiedEulerMethod(
                pde_solver=pde_solver,)
            system = cls(
                network=network,
                extender=extender,
                exp_name=input_file,
            )

        return system

    def __update_growth_gauges(self, dt):
        """Update growth gauges."""
        self.growth_gauges[1], self.growth_gauges[2] = self.network.height_and_length()
        if not (isinstance(dt, list) or isinstance(dt, np.ndarray)):
            dt = [dt]
        self.growth_gauges[0] = self.growth_gauges[0] + len(dt)-1
        for dt_i in dt:
            self.growth_gauges[3] = self.growth_gauges[3] + dt_i
            self.timestamps = np.append( self.timestamps, self.growth_gauges[3] )

        print("Active branches: {n:d}".format(
            n=len(self.network.active_branches)))
        print("Network height: {h:.3f}".format(h=self.growth_gauges[1]))
        print("Network length: {l:.3f}".format(l=self.growth_gauges[2]))
        print("Evolution time: {t:.3f}".format(t=self.growth_gauges[3]))

    def evolve(self, ax=None):
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
            
            # morphing the system: jellyfish, leaf
            if self.morpher is not None:
                # ax.plot(*out_growth[1][:,:2].T, '.-', ms=5, color="tab:orange")
                out_growth = self.morpher.morph(network=self.network, \
                                                out_growth=out_growth, \
                                                step=self.growth_gauges[0])
            
            
            # Updating gauges, etc.
            self.__update_growth_gauges(out_growth[0])
            t_diff = time.time() - start_clock
            print(f"Computation time: {int(t_diff/3600):d}h {int((t_diff%3600)/60):d}min")
            
            if not self.growth_gauges[0] % self.dump_every:
                self.export_json()
                if ax is not None:
                    # ax.clear()
                    # graphics.plot_tree(ax, self)
                    ax.plot(*self.network.box.points.T, \
                            '.-', ms=2, \
                            color="tab:red", lw=0.5, alpha=0.5)
                    ax.plot(*self.network.branches[0].points.T, color="darkred", alpha=0.5)
                    plt.pause(0.01)
                    
        self.export_json()
        print("\n End of the simulation")

if __name__ == "__main__":
    box, branches, active_branches = Box.construct(
        initial_condition=100
    )