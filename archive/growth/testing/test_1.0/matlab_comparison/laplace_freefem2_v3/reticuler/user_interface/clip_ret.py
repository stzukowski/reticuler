"""Command line script to clip a network"""

import argparse
import json
import textwrap
import matplotlib.pyplot as plt

from reticuler.system import System
from reticuler.user_interface import graphics, clippers

# %%
def main():
    parser = argparse.ArgumentParser(
        description="Clip a network.", formatter_class=argparse.RawTextHelpFormatter
    )

    # defining arguments for parser object
    parser.add_argument(
        "input_file",
        type=str,
        nargs=1,
        metavar="file_name",
        help=textwrap.dedent(
            """\
                            File to import"""
        ),
    )
    parser.add_argument(
        "-out",
        "--output_file",
        type=str,
        nargs=1,
        metavar="file_name",
        help=textwrap.dedent(
            """\
                            File to export. If None the same as input (+"_clipped" at the end).
                            default = None """
        ),
        default=None,
    )

    # Clipping type and limit
    parser.add_argument(
        "-S",
        "--step",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent(
            """\
                            Maximum step in the network evolution. If filled clips to step.
                            default = None
                            """
        ),
        default=None,
    )
    parser.add_argument(
        "-L",
        "--length",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent(
            """\
                            Maximum length of the network. If filled clips to length.
                            default = None
                            """
        ),
        default=None,
    )
    parser.add_argument(
        "-H",
        "--height",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent(
            """\
                            Maximum height of the network. If filled clips to height.
                            default = None
                            """
        ),
        default=None,
    )

    # parse the arguments from standard input
    args = parser.parse_args()

    # Import System from JSON file
    system = System.import_json(input_file=args.input_file[0])
    
    if args.step is not None:
        clippers.clip_to_step(system, args.step[0])
    elif args.length is not None:
        clippers.clip_to_length(system, args.length[0])
    elif args.height is not None:
        clippers.clip_to_height(system, args.height[0])
    else:
        print("Network not clipped - you must choose one clipping limit!")

    if args.output_file is None:
        system.exp_name = args.input_file[0] + "_clipped"
    else:
        system.exp_name = args.output_file[0]
    system.export_json()


if __name__ == "__main__":
    main()
