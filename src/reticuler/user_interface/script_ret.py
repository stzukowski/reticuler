"""Command line script to prepare FreeFEM scripts based on the system"""

import glob
import argparse
import textwrap

from reticuler.system import System

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a FreeFEM script based on the system.", formatter_class=argparse.RawTextHelpFormatter
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
                            File to export. If None the same as input+`.edp`.
                            default = None """
        ),
        default=None,
    )
        
    # Prepare script for all files in the directory
    parser.add_argument(
        "-all",
        "--script_all",
        action=argparse.BooleanOptionalAction,
        help=textwrap.dedent(
            """\
            Flag indicating to prepare scripts from all files in the directory.
            """
        ),
    )    
    
    # parse the arguments from standard input
    args = parser.parse_args()
    
    if args.script_all:
        file_names = glob.glob(args.input_file[0] + "*.json")
    else:
        file_names = [ args.input_file[0]+".json" ]
    
    for file in file_names:
        exp_name = file[:-5]
        # Import System from JSON file
        system = System.import_json(input_file=exp_name)

        if len(system.network.active_branches)==0:
            print("Cannot create the script - no active tips. Clip network.")
        else:
            script = system.extender.pde_solver.prepare_script(system.network)
            if args.output_file is None:
                filename = exp_name + "_script.edp"
            else:
                filename = args.output_file[0] + ".edp"
                
            with open(filename, "w") as edp_temp_file:
                edp_temp_file.write(script)

if __name__ == "__main__":
    main()
