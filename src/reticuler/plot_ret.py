import argparse
import json
import textwrap
import matplotlib.pyplot as plt

from reticuler.system import System
from reticuler.user_interface import graphics

# %% 
def main():
    parser = argparse.ArgumentParser(description = 'Grow a network.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    # defining arguments for parser object
    parser.add_argument('input_file', type = str, nargs = 1,
                        metavar = 'file_name',
                        help = textwrap.dedent('''\
                            File to import''') )
    parser.add_argument('-out', '--output_file', type = str, nargs = 1,
                        metavar = 'file_name',
                        help = textwrap.dedent('''\
                            File to export. If None the same as input.
                            default = '' '''),
                        default = None)
                            
    parser.add_argument('-out_ext', '--output_extension', type = str, nargs = 1,
                        metavar = 'ext',
                        help = textwrap.dedent('''\
                            Output extension ('.pdf', '.svg', '.png', etc.)
                            default = '.pdf' '''),
                        default = ['.pdf'])                            
    
    # Plotting options
    parser.add_argument('--ylim', type = float, nargs = 1,
                        metavar = 'num', 
                        help = textwrap.dedent('''\
                            ylim of the plot. If None ylim=max height of the network.
                            default = None
                            '''),
                        default = [None])
    parser.add_argument('--xlim', type = float, nargs = 1,
                        metavar = 'num', 
                        help = textwrap.dedent('''\
                            xlim of the plot.
                            default = 2
                            '''),
                        default = [2])                        
    parser.add_argument('--plot_params', type = json.loads, nargs = 1,
                        metavar = 'dict', 
                        help = textwrap.dedent('''\
                            Optional plotting parameters.
                            
                            Pass dictionary in a form (no spaces!): "{\"value\":key}"
                            default = {} (keeps default values as listed below)
                            
                            ''') + textwrap.dedent(graphics.plot_tree.__doc__[\
                                    graphics.plot_tree.__doc__.find('kwargs_plots')-4:\
                                        graphics.plot_tree.__doc__.find('Returns')]),
                        default = [{}])
    
    # parse the arguments from standard input
    args = parser.parse_args()

    # Import System from JSON file
    system = System.import_json(input_file=args.input_file[0])

    fig, ax = plt.subplots()
    graphics.plot_tree( ax, network=system.network, ylim=args.ylim[0], xlim=args.xlim[0], **args.plot_params[0] )
    
    if args.output_file is None:
        fig.savefig(args.input_file[0]+args.output_extension[0], bbox_inches='tight')
    else:
        fig.savefig(args.output_file[0]+args.output_extension[0], bbox_inches='tight')
    
if __name__ == '__main__':
    main()