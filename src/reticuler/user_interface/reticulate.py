"""Command line script to run the simulation"""

import logging
import argparse
import json
import textwrap
import importlib.metadata

from reticuler.utilities.geometry import Box
from reticuler.system import System
from reticuler.extending_kernels import extenders, pde_solvers
from reticuler.utilities import morphers
from reticuler.user_interface import runner


# %%
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
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
        help=textwrap.dedent("""\
            File to import. If None, the System is prepared based on the rest of the arguments.
            default = None"""),
        default=[None],
    )

    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        nargs=1,
        metavar="exp_name",
        help=textwrap.dedent("""\
            File to export. If we import a file and leave this as default,
            ``system.exp_name`` will be set to ``input_file``.
            default = None"""),
        default=[None],
    )

    # Growth options
    parser.add_argument(
        "--growth_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
            Optional growth parameters.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around ``value``): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """)
        + textwrap.dedent(
            System.__doc__[
                System.__doc__.find("growth_thresh_type") : System.__doc__.find(
                    "exp_name"
                )
            ]
        ),
        default=[None],
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
                Box.construct.__doc__.find(
                    "initial_condition"
                ) : Box.construct.__doc__.find("kwargs_construct")
            ]
        ),
        default=[None],
    )
    parser.add_argument(
        "--kwargs_box",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
            Kwargs for Box construct method.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around ``value``): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """)
        + textwrap.dedent(
            Box.construct.__doc__[
                Box.construct.__doc__.find(
                    "kwargs_construct"
                ) : Box.construct.__doc__.find("Returns")
                - 2
            ]
        ),
        default=[None],
    )

    # Solver
    parser.add_argument(
        "--pde_solver",
        type=str,
        nargs=1,
        metavar="name",
        help=textwrap.dedent("""\
            PDE solver
            default = FreeFEM_ThinFingers"""),
        default=[None],
    )
    parser.add_argument(
        "--pde_solver_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
            Optional parameters for solver.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around ``value``): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """)
        + "1. FreeFEM_ThinFingers\n"
        + textwrap.dedent(
            pde_solvers.FreeFEM_ThinFingers.__doc__[
                pde_solvers.FreeFEM_ThinFingers.__doc__.find(
                    "equation"
                ) : pde_solvers.FreeFEM_ThinFingers.__doc__.find("References")
                - 2
            ]
        )
        + "\n\n\n2. FreeFEM_ThinFingers_Boundary\n"
        + textwrap.dedent(
            pde_solvers.FreeFEM_ThinFingers_Boundary.__doc__[
                pde_solvers.FreeFEM_ThinFingers_Boundary.__doc__.find(
                    "equation"
                ) : pde_solvers.FreeFEM_ThinFingers_Boundary.__doc__.find("References")
                - 2
            ]
        )
        + "\n\n\n3. FreeFEM_ThickFingers\n"
        + textwrap.dedent(
            pde_solvers.FreeFEM_ThickFingers.__doc__[
                pde_solvers.FreeFEM_ThickFingers.__doc__.find(
                    "equation"
                ) : pde_solvers.FreeFEM_ThickFingers.__doc__.find("References")
                - 2
            ]
        )
        + "\n\n\n4. FreeFEM_ThickFingers_Elasticity\n"
        + textwrap.dedent(
            pde_solvers.FreeFEM_ThickFingers_Elasticity.__doc__[
                pde_solvers.FreeFEM_ThickFingers_Elasticity.__doc__.find(
                    "equation"
                ) : pde_solvers.FreeFEM_ThickFingers_Elasticity.__doc__.find(
                    "References"
                )
                - 2
            ]
        ),
        default=[None],
    )

    # Extender
    parser.add_argument(
        "--extender",
        type=str,
        nargs=1,
        metavar="name",
        help=textwrap.dedent("""\
            Extender
            default = ModifiedEulerMethod"""),
        default=[None],
    )
    parser.add_argument(
        "--extender_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
            Optional parameters for extender.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around ``value``): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """)
        + "1. ModifiedEulerMethod\n"
        + textwrap.dedent(
            extenders.ModifiedEulerMethod.__doc__[
                extenders.ModifiedEulerMethod.__doc__.find(
                    "is_reconnecting"
                ) : extenders.ModifiedEulerMethod.__doc__.find("References")
                - 4
            ]
        ),
        default=[None],
    )

    # Morpher
    parser.add_argument(
        "--morpher",
        type=str,
        nargs=1,
        metavar="name",
        help=textwrap.dedent("""\
        Morpher
        default = None"""),
        default=[None],
    )
    parser.add_argument(
        "--morpher_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
            Optional parameters for morpher.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around ``value``): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """)
        + "1. Jellyfish\n"
        + textwrap.dedent(
            morphers.Jellyfish.__doc__[morphers.Jellyfish.__doc__.find("radii") :]
        ),
        default=[None],
    )

    # Plotting at the end
    parser.add_argument(
        "-fp",
        "--final_plot",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent("""\
            Flag indicating to plot the final network.
            """),
    )

    # parse the arguments from standard input
    args = parser.parse_args(argv)


    raw_params = {
        "output_file": args.output_file[0],
        "growth_params": args.growth_params[0],
        "initial_condition": args.initial_condition[0],
        "kwargs_box": args.kwargs_box[0],
        "pde_solver": args.pde_solver[0],
        "pde_solver_params": args.pde_solver_params[0],
        "extender": args.extender[0],
        "extender_params": args.extender_params[0],
        "morpher": args.morpher[0],
        "morpher_params": args.morpher_params[0],
        "input_file": args.input_file[0],
    }
    params = {k: v for k, v in raw_params.items() if v}  # drops unset/empty entries so
    # System.construct's own defaults apply for anything not given here.
    params["final_plot"] = bool(args.final_plot)  # set after the filter: False must survive

    runner.run_experiment(params, log_to_file=False)


if __name__ == "__main__":
    main()
