"""Command line script to plot a network"""

import glob
import argparse
import json
import textwrap
import matplotlib.pyplot as plt
import numpy as np

from reticuler.system import System
from reticuler.user_interface import graphics, clippers
from reticuler.utilities.misc import create_dir

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
                            File to export. If None the same as input.
                            default = None """
        ),
        default=None,
    )

    parser.add_argument(
        "-out_ext",
        "--output_extension",
        type=str,
        nargs=1,
        metavar="ext",
        help=textwrap.dedent(
            """\
                            Output extension (".pdf", ".svg", ".png", etc.)
                            default = ".jpg" """
        ),
        default=[".jpg"],
    )

    # Plotting options
    parser.add_argument(
        "-X",
        "--xmax",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent(
            """\
                            xlim of the plot. If None xmax=max x of the box.
                            default = None
                            """
        ),
        default=[None],
    )
    parser.add_argument(
        "-Y",
        "--ymax",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent(
            """\
                            ymax of the plot. If None ymax=max y of the network.
                            default = None
                            """
        ),
        default=[None],
    )
    parser.add_argument(
        "--plot_params",
        type=json.loads,
        nargs=1,
        metavar="dict",
        help=textwrap.dedent(
            """\
                            Optional plotting parameters.
                            
                            Pass dictionary in a form (no spaces, 
                            backslash before quotes around `value`): 
                                "{\\"value\\":key}"
                            default = {} (keeps default values as listed below)
                            
                            """
        )
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
        help=textwrap.dedent(
            """\
            Flag indicating to plot networks from all the files in the directory.
            """
        ),
    )    

    # Rotate the plot
    parser.add_argument(
        "-rot",
        "--rot_angle",        
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent(
            """\
            An angle by which the output will be rotated.
            default = 0
            """
        ),
        default=[None],
    )

    # Save frames with each step
    parser.add_argument(
        "-f",
        "--frames",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent(
            """\
            Flag indicating to plot all frames.
            """
        ),
    )        
    # Animate the growth
    parser.add_argument(
        "-anim",
        "--animate",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent(
            """\
            Flag indicating to animate the growth of a network.
            """
        ),
    )
    # Speed factor for the animation
    parser.add_argument(
        "-speed",
        "--speed_factor",
        type=float,
        nargs=1,
        metavar="num",
        help=textwrap.dedent(
            """\
            A speed factor by which the animation will be accelerated.
            default = 1 (animation length = 4s)
            """
        ),
        default=[None],
    )        
        

    # parse the arguments from standard input
    args = parser.parse_args()
    
    if args.plot_all:
        file_names = glob.glob(args.input_file[0] + "*.json")
    else:
        file_names = [ args.input_file[0]+".json" ]
    
    for file in file_names:
        exp_name = file[:-5]
        # Import System from JSON file
        system0 = System.import_json(input_file=exp_name)
    
        if args.animate:
            ani = graphics.animate_tree(
                system0=system0,
                xmax=args.xmax[0],
                ymax=args.ymax[0],
                speed_factor=args.speed_factor[0],
                rot_angle=args.rot_angle[0],
                **args.plot_params[0]
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
                steps = np.arange(system0.growth_gauges[0]+1, dtype=int)
                create_dir(f"frames_{exp_name}/")
                output_name = f"frames_{exp_name}/" + output_name
             
            for s in steps:
                print(f"Plotting step {s}")
                system = system0.copy()
                if args.frames:
                    clippers.clip_to_step(system, s)
                fig, ax = plt.subplots()
                graphics.plot_tree(
                    ax,
                    system=system,
                    xmax=args.xmax[0],
                    ymax=args.ymax[0],
                    rot_angle=args.rot_angle[0],
                    **args.plot_params[0]
                )
            
                fig.savefig(output_name + f"_S{s}" + args.output_extension[0], 
                            bbox_inches="tight", dpi=400)


if __name__ == "__main__":
    main()
