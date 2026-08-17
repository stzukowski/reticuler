"""Command line script to plot a network"""

import glob
import logging
import argparse
import json
import re
import textwrap
import matplotlib.pyplot as plt
import numpy as np

from reticuler.system import System
from reticuler.user_interface import graphics, clippers
from reticuler.utilities.misc import create_dir
from reticuler.backward_evolution.postprocessor import BEAPostProcessor

logger = logging.getLogger("reticuler")

def plot_back(args):
    """Handle `-back`: plot a BEA eta* scan.

    `file_name` is used as a prefix, same as with `-all`: every
    "<file_name>eta##_back.json" file (as produced by `run_BEA.py`) is read
    to infer eta_range, then plotted against "original_tree.json" (expected
    in the current directory) via BEAPostProcessor.
    """
    system0 = System.import_json(input_file="original_tree")

    prefix = args.input_file[0]
    pattern = re.compile(re.escape(prefix.replace("\\", "/")) + r"eta(\d+)_back\.json$")
    etas = [
        int(m.group(1)) / 10
        for f in glob.glob(prefix + "eta*_back.json")
        if (m := pattern.match(f.replace("\\", "/")))
    ]
    if not etas:
        raise FileNotFoundError(f"No '{prefix}eta##_back.json' files found.")
    eta_range = np.array(sorted(etas))

    output_name = prefix + "BEA_results"
    if args.output_file is not None:
        output_name = args.output_file[0]

    logger.info("Plotting BEA scan %s (eta_range=%s)", prefix, eta_range)
    post = BEAPostProcessor(
        exp_name=prefix, system=system0, eta_range=eta_range, **args.back_params[0]
    )
    fig, _ = post.plot()
    fig.savefig(output_name + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a network.", formatter_class=argparse.RawTextHelpFormatter
    )

    # defining arguments for parser object
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
        metavar="file_name",
        help=textwrap.dedent("""\
                            File to export. If None the same as input.
                            default = None """),
        default=None,
    )

    parser.add_argument(
        "-out_ext",
        "--output_extension",
        type=str,
        nargs=1,
        metavar="ext",
        help=textwrap.dedent("""\
                            Output extension (".pdf", ".svg", ".png", etc.)
                            default = ".jpg" """),
        default=[".jpg"],
    )

    # Plotting options
    parser.add_argument(
        "--xmin",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent("""\
                            xlim of the plot. If None xmin=min x of the box.
                            default = None
                            """),
        default=[None],
    )
    parser.add_argument(
        "--xmax",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent("""\
                            xlim of the plot. If None xmin=max x of the box.
                            default = None
                            """),
        default=[None],
    )
    parser.add_argument(
        "--ymin",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent("""\
                            ymax of the plot. If None ymin=min y of the network.
                            default = None
                            """),
        default=[None],
    )
    parser.add_argument(
        "--ymax",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent("""\
                            ymax of the plot. If None ymax=max y of the network.
                            default = None
                            """),
        default=[None],
    )
    parser.add_argument(
        "--plot_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
                            Optional plotting parameters.
                            
                            Pass dictionary in a form (no spaces, 
                            backslash before quotes around `value`): 
                                "{\\"value\\":key}"
                            default = {} (keeps default values as listed below)
                            
                            """)
        + textwrap.dedent(
            graphics.plot_tree.__doc__[
                graphics.plot_tree.__doc__.find("kwargs_plots")
                - 4 : graphics.plot_tree.__doc__.find("Returns")
            ]
        ),
        default=[{}],
    )

    # Plot all files in the directory
    parser.add_argument(
        "-all",
        "--plot_all",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent("""\
            Flag indicating to plot networks from all the files in the directory.
            """),
    )

    # Plot a BEA eta* scan
    parser.add_argument(
        "-back",
        "--backward",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent("""\
            Flag indicating to plot a BEA results. `file_name` is used as a prefix,
            same as with -all: every "<file_name>eta##_back.json" file is
            read to infer eta_range, then plotted against
            "original_tree.json" (expected in the current directory) via
            BEAPostProcessor.
            """),
    )
    parser.add_argument(
        "-back_params",
        "--back_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent("""\
                            Optional parameters for -back, passed to BEAPostProcessor
                            (e.g. eta_original, bif_type).

                            Pass dictionary in a form (no spaces,
                            backslash before quotes around `value`):
                                "{\\"value\\":key}"
                            default = {} (keeps BEAPostProcessor's own defaults)

                            """),
        default=[{}],
    )

    # Rotate the plot
    parser.add_argument(
        "-rot",
        "--rot_angle",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent("""\
            An angle by which the output will be rotated.
            default = 0
            """),
        default=[None],
    )

    # Save frames with each step
    parser.add_argument(
        "-f",
        "--frames",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent("""\
            Flag indicating to plot all frames.
            """),
    )
    # Animate the growth
    parser.add_argument(
        "-anim",
        "--animate",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent("""\
            Flag indicating to animate the growth of a network.
            """),
    )
    # Speed factor for the animation
    parser.add_argument(
        "-speed",
        "--speed_factor",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent("""\
            A speed factor by which the animation will be accelerated.
            default = 1 (animation length = 4s)
            """),
        default=[None],
    )

    # parse the arguments from standard input
    args = parser.parse_args()

    if args.backward:
        plot_back(args)
        return

    if args.plot_all:
        file_names = glob.glob(args.input_file[0] + "*.json")
    else:
        file_names = [args.input_file[0] + ".json"]

    for file in file_names:
        exp_name = file[:-5]
        # Import System from JSON file
        system0 = System.import_json(input_file=exp_name)

        if args.animate:
            ani = graphics.animate_tree(
                system0=system0,
                xmin=args.xmin[0],
                xmax=args.xmax[0],
                ymin=args.ymin[0],
                ymax=args.ymax[0],
                speed_factor=args.speed_factor[0],
                rot_angle=args.rot_angle[0],
                **args.plot_params[0],
            )
            if args.output_file is None:
                ani.save(exp_name + ".avi", writer="ffmpeg", dpi=600)
            else:
                ani.save(args.output_file[0] + ".avi", writer="ffmpeg", dpi=600)
        else:
            output_name = exp_name
            if args.output_file is not None:
                output_name = args.output_file[0]

            steps = [int(system0.growth_gauges[0])]
            if args.frames:
                steps = np.arange(system0.growth_gauges[0] + 1, dtype=int)
                create_dir(f"frames_{exp_name}/")
                output_name = f"frames_{exp_name}/" + output_name

            for s in steps:
                logger.info("Plotting %s, step %d", exp_name, s)
                system = system0.copy()
                if args.frames:
                    clippers.clip_to_step(system, s)
                fig, ax = plt.subplots()
                graphics.plot_tree(
                    system=system,
                    ax=ax,
                    xmin=args.xmin[0],
                    xmax=args.xmax[0],
                    ymin=args.ymin[0],
                    ymax=args.ymax[0],
                    rot_angle=args.rot_angle[0],
                    **args.plot_params[0],
                )
                if args.frames:
                    fn = output_name + f"_S{s}" + args.output_extension[0]
                else:
                    fn = output_name + args.output_extension[0]
                fig.savefig(fn, bbox_inches="tight", dpi=400)
                plt.close()


if __name__ == "__main__":
    main()
