"""Command line script to run the simulation"""

import argparse
import json
import textwrap
import importlib.metadata
import matplotlib.pyplot as plt
import numpy as np

from reticuler.utilities.geometry import Box, Network
from reticuler.system import System
from reticuler.extending_kernels import extenders, pde_solvers
from reticuler.utilities import morphers
from reticuler.user_interface import graphics

# %%
def main():
    parser = argparse.ArgumentParser(
        description="Grow a network.", formatter_class=argparse.RawTextHelpFormatter
    )

    # defining arguments for parser object
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("reticuler"),
    )

    parser.add_argument(
        "-in",
        "--input_file",
        type=str,
        nargs=1,
        metavar="exp_name",
        help=textwrap.dedent(
            """\
            File to import. If None, the System is prepared based on the rest of the arguments.
            default = None"""
        ),
        default=None,
    )

    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        nargs=1,
        metavar="exp_name",
        help=textwrap.dedent(
            """\
            File to export. If we import a file and leave this as default, 
            `system.exp_name` will be set to `input_file`.
            default = '' """
        ),
        default=[""],
    )

    # Growth options
    parser.add_argument(
        "--growth_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent(
            """\
            Optional growth parameters.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around `value`): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """
        )
        + textwrap.dedent(
            System.__doc__[
                System.__doc__.find("growth_thresh_type")
                - 4 : System.__doc__.find("exp_name")
            ]
        ),
        default=[{}],
    )

    # Box options
    parser.add_argument(
        "-ic",
        "--initial_condition",
        type=int,
        nargs=1,
        metavar="label",
        help=textwrap.dedent(
            Box.construct.__doc__[
                Box.construct.__doc__.find("initial_condition")
                - 8 : Box.construct.__doc__.find("kwargs_construct")
            ]
        ),
        default=[0],
    )
    parser.add_argument(
        "--kwargs_box",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent(
            """\
            Kwargs for Box construct method.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around `value`): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """
        )
        + textwrap.dedent(
            Box.construct.__doc__[
                Box.construct.__doc__.find("kwargs_construct")
                - 8 : Box.construct.__doc__.find("Returns")
            ]
        ),
        default=[{}],
    )

    # Solver
    parser.add_argument(
        "--pde_solver",
        type=str,
        nargs=1,
        metavar="name",
        help=textwrap.dedent(
            """\
            PDE solver
            default = FreeFEM"""
        ),
        default=["FreeFEM"],
    )
    parser.add_argument(
        "--pde_solver_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent(
            """\
            Optional parameters for solver.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around `value`): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """
        )
        + "1. FreeFEM\n"
        + textwrap.dedent(
            pde_solvers.FreeFEM.__doc__[
                pde_solvers.FreeFEM.__doc__.find("equation")
                - 4 : pde_solvers.FreeFEM.__doc__.find("flux_info")
            ]
        )
        + "2. FreeFEM_ThickFingers\n"
        + textwrap.dedent(
            pde_solvers.FreeFEM_ThickFingers.__doc__[
                pde_solvers.FreeFEM_ThickFingers.__doc__.find("equation")
                - 4 : pde_solvers.FreeFEM_ThickFingers.__doc__.find("flux_info")
            ]
        ),
        default=[{}],
    )

    # Extender
    parser.add_argument(
        "--extender",
        type=str,
        nargs=1,
        metavar="name",
        help=textwrap.dedent(
            """\
            Extender
            default = ModifiedEulerMethod"""
        ),
        default=["ModifiedEulerMethod"],
    )
    parser.add_argument(
        "--extender_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent(
            """\
            Optional parameters for extender.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around `value`): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """
        )
        + "1. ModifiedEulerMethod\n"
        + textwrap.dedent(
            extenders.ModifiedEulerMethod.__doc__[
                extenders.ModifiedEulerMethod.__doc__.find("eta")
                - 4 : extenders.ModifiedEulerMethod.__doc__.find("References")
            ]
        ),
        default=[{}],
    )

    # Plotting at the end
    parser.add_argument(
        "-fp",
        "--final_plot",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent(
            """\
            Flag indicating to plot the final network.
            """
        ),
    )

    # parse the arguments from standard input
    args = parser.parse_args()

    if args.input_file is None:
        # Prepare the System from scratch

        # Box
        box, branches, active_branches, branch_connectivity = Box.construct(
            initial_condition=args.initial_condition[0], **args.kwargs_box[0]
        )

        # Network
        network = Network(box=box, branches=branches, 
                          active_branches=active_branches, 
                          branch_connectivity=branch_connectivity)
        
        # Morpher
        if args.initial_condition[0]==4 or args.initial_condition[0]==5:
            morpher = morphers.Jellyfish(
                        radii=np.array([(network.box.points[:,0].min()+network.box.points[:,0].max())/2])
                        )
        elif args.initial_condition[0]==6:
            morpher = morphers.Leaf(box_history=[box.copy()])
        else:
            morpher = None
          
        # Extender
        if args.extender[0] == "ModifiedEulerMethod":
            # Solver
            if args.pde_solver[0] == "FreeFEM":
                pde_solver = pde_solvers.FreeFEM(network, **args.pde_solver_params[0])
            elif args.pde_solver[0] == "FreeFEM_ThickFingers":
                pde_solver = pde_solvers.FreeFEM_ThickFingers(network, **args.pde_solver_params[0])  
            
            extender = extenders.ModifiedEulerMethod(
                pde_solver=pde_solver, **args.extender_params[0]
            )          

        # General
        system = System(
            network=network,
            extender=extender,
            morpher=morpher,
            exp_name=args.output_file[0],
            **args.growth_params[0]
        )

    else:
        # Import System from JSON file
        system = System.import_json(input_file=args.input_file[0])
        if args.output_file[0] != "":
            system.exp_name = args.output_file[0]

    system.evolve()

    if args.final_plot:
        fig, ax = plt.subplots()
        graphics.plot_tree(ax, system=system, ymin=0)
        fig.savefig(system.exp_name + ".jpg", bbox_inches="tight",dpi=300)
        
        # ani = graphics.animate_tree(system0=system)
        # ani.save(system.exp_name + ".avi", writer="ffmpeg", dpi=600)


if __name__ == "__main__":
    main()
