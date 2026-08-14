"""Command line script to run the Backward Evolution Algorithm"""

import logging
import argparse
import json
import textwrap
import importlib.metadata

from reticuler.backward_evolution.system_back import BackwardSystem
from reticuler.backward_evolution import trimmers


# %%
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Run the Backward Evolution Algorithm.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # defining arguments for parser object
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("reticuler"),
    )
    parser.add_argument(
        "input_file",
        type=str,
        nargs=1,
        metavar="file_name",
        help=textwrap.dedent("""\
                            File to import"""),
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        nargs=1,
        metavar="exp_name",
        help=textwrap.dedent("""\
            File to export:
            exp_name = ``exp_name``+'_back.json'.
            If left as default:
            exp_name = ``input_file``+'_back.json'.

            default = None
            """),
        default=[None],
    )

    # BEA options
    parser.add_argument(
        "--BEA_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
            Optional BEA parameters.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around `value`): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """)
        + textwrap.dedent(
            BackwardSystem.__doc__[
                BackwardSystem.__doc__.find("BEA_step_thresh")
                - 4 : BackwardSystem.__doc__.find("exp_name")
            ]
        ),
        default=[None],
    )

    # Trimmer
    parser.add_argument(
        "--trimmer",
        type=str,
        nargs=1,
        metavar="name",
        help=textwrap.dedent("""\
            Trimmer
            default = BackwardModifiedEulerMethod"""),
        default=["BackwardModifiedEulerMethod"],
    )
    parser.add_argument(
        "--trimmer_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
            Optional parameters for trimmer.
            
            Pass dictionary in a form (no spaces, 
            backslash before quotes around `value`): 
                "{\"value\":key}"
            default = {} (keeps default values as listed below)
            
            """)
        + textwrap.dedent(
            trimmers.BackwardModifiedEulerMethod.__doc__[
                trimmers.BackwardModifiedEulerMethod.__doc__.find("eta")
                - 4 : trimmers.BackwardModifiedEulerMethod.__doc__.find("References")
                - 6
            ]
        ),
        default=[None],
    )

    # Continuation
    parser.add_argument(
        "-cont",
        "--continuation_file",
        type=str,
        nargs=1,
        metavar="file_name",
        help=textwrap.dedent("""\
            Continuation of the previously commenced BEA.
            System will be imported from ``input_file`` and backward system from 
            ``continuation_file``+'.json'.
            """),
        default=[None],
    )

    # parse the arguments from standard input
    args = parser.parse_args(argv)

    raw_params = {
        "input_file": args.input_file[0],
        "output_file": args.output_file[0],
        "BEA_params": args.BEA_params[0],
        "trimmer_params": args.trimmer_params[0],
        "continuation_file": args.continuation_file[0],
    }
    backward_system = BackwardSystem.construct(
        {k: v for k, v in raw_params.items() if v}
    )

    # Running BEA
    backward_system.run_BEA()

    return backward_system


if __name__ == "__main__":
    main()
