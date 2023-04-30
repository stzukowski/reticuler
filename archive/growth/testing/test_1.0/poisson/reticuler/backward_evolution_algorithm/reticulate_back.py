"""Command line script to run the Backward Evolution Algorithm"""

import argparse
import json
import textwrap
import importlib.metadata

from reticuler.system import System
from reticuler.backward_evolution_algorithm.system_back import BackwardEvolutionAlgorithm
from reticuler.backward_evolution_algorithm import trimmers

# %%
def main():
    parser = argparse.ArgumentParser(
        description="Run the Backward Evolution Algorithm.", formatter_class=argparse.RawTextHelpFormatter
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
            File to export.
            default = '' """
        ),
        default=[""],
    )

    # BEA options
    parser.add_argument(
        "--BEA_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent(
            """\
            Optional BEA parameters.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around `value`): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """
        )
        + textwrap.dedent(
            BackwardEvolutionAlgorithm.__doc__[
                BackwardEvolutionAlgorithm.__doc__.find("BEA_step_thresh")
                - 4 : BackwardEvolutionAlgorithm.__doc__.find("exp_name")
            ]
        ),
        default=[{}],
    )

    # Trimmer
    parser.add_argument(
        "--trimmer",
        type=str,
        nargs=1,
        metavar="name",
        help=textwrap.dedent(
            """\
            Trimmer
            default = BackwardModifiedEulerMethod"""
        ),
        default=["BackwardModifiedEulerMethod"],
    )
    parser.add_argument(
        "--trimmer_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent(
            """\
            Optional parameters for trimmer.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around `value`): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """
        )
        + textwrap.dedent(
            trimmers.BackwardModifiedEulerMethod.__doc__[
               trimmers.BackwardModifiedEulerMethod.__doc__.find("eta")
                - 4 : trimmers.BackwardModifiedEulerMethod.__doc__.find("References")-6
            ]
        ),
        default=[{}],
    )

    # parse the arguments from standard input
    args = parser.parse_args()

    # Import System from JSON file
    system = System.import_json(input_file=args.input_file[0])
    if args.output_file[0] != "":
        system.exp_name = args.output_file[0]

    # Trimmer
    if args.trimmer[0] == "BackwardModifiedEulerMethod":
        trimmer = trimmers.BackwardModifiedEulerMethod(system.extender.pde_solver, **args.trimmer_params[0])
        
    # General
    BEA = BackwardEvolutionAlgorithm(system, trimmer)
    BEA.run()

if __name__ == "__main__":
    main()
