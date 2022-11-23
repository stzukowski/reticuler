"""Command line script to run the simulation"""

import argparse
import json
import textwrap
import matplotlib.pyplot as plt

from reticuler.system import Box, Network, System
from reticuler.extending_kernels import extenders, pde_solvers, trajectory_integrators
from reticuler.user_interface import graphics

# %% 
def main():
    parser = argparse.ArgumentParser(description = 'Grow a network.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    # defining arguments for parser object
    parser.add_argument('-in', '--input_file', type = str, nargs = 1,
                        metavar = 'exp_name',
                        help = textwrap.dedent('''\
                            File to import. If None, the System is prepared based on the rest of the arguments.
                            default = None'''),
                        default = None)
    parser.add_argument('-out', '--output_file', type = str, nargs = 1,
                        metavar = 'exp_name',
                        help = textwrap.dedent('''\
                            File to export.
                            default = '' '''),
                        default = [''])
    
    # Growth options
    parser.add_argument('--growth_params', type = json.loads, nargs = 1,
                        metavar = 'dict', 
                        help = textwrap.dedent('''\
                            Optional growth parameters.
                            
                            Pass dictionary in a form (no spaces, 
                            backslash before quotes around `value`): 
                                "{\"value\":key}"
                            default = {} (keeps default values as listed below)
                            
                            ''') + textwrap.dedent(System.__doc__[\
                                    System.__doc__.find('growth_thresh_type')-4:\
                                        System.__doc__.find('exp_name')]),
                        default = [{}])
    
    # Box options
    parser.add_argument('-ic', '--initial_condition', type = int, nargs = 1,
                        metavar = 'label', 
                        help = textwrap.dedent(Box.construct.__doc__[\
                                Box.construct.__doc__.find('initial_condition')-8:\
                                    Box.construct.__doc__.find('kwargs_construct')]),
                        default = [0])
    parser.add_argument('--kwargs_box', type = json.loads, nargs = 1,
                        metavar = 'dict', 
                        help = textwrap.dedent('''\
                            Kwargs for Box construct method.
                            
                            Pass dictionary in a form (no spaces, 
                            backslash before quotes around `value`): 
                                "{\"value\":key}"
                            default = {} (keeps default values as listed below)
                            
                            ''') + textwrap.dedent(Box.construct.__doc__[\
                                    Box.construct.__doc__.find('kwargs_construct')-8:\
                                        Box.construct.__doc__.find('Returns')]),
                        default = [{}])

    # Trajectory integrator
    parser.add_argument('--trajectory_integrator', type = str, nargs = 1,
                        metavar = 'name', 
                        help = textwrap.dedent('''\
                            Trajectory integrator
                            default = modified_euler'''),
                        default = ['modified_euler'])
    
    # Solver
    parser.add_argument('--pde_solver', type = str, nargs = 1,
                        metavar = 'name', 
                        help = textwrap.dedent('''\
                            PDE solver
                            default = FreeFEM'''),
                        default = ['FreeFEM'])
    parser.add_argument('--pde_solver_params', type = json.loads, nargs = 1,
                        metavar = 'dict', 
                        help = textwrap.dedent('''\
                            Optional parameters for solver.
                            
                            Pass dictionary in a form (no spaces, 
                            backslash before quotes around `value`): 
                                "{\"value\":key}"
                            default = {} (keeps default values as listed below)
                            
                            ''') + \
                            '1. FreeFEM\n' + textwrap.dedent(pde_solvers.FreeFEM.__doc__[\
                                    pde_solvers.FreeFEM.__doc__.find('equation')-4:\
                                        pde_solvers.FreeFEM.__doc__.find('a1a2a3_coefficients')]),
                        default = [{}])                            
        
    # Extender
    parser.add_argument('--extender', type = str, nargs = 1,
                        metavar = 'name', 
                        help = textwrap.dedent('''\
                            Extender
                            default = Streamline'''),
                        default = ['Streamline'])
    parser.add_argument('--extender_params', type = json.loads, nargs = 1,
                        metavar = 'dict', 
                        help = textwrap.dedent('''\
                            Optional parameters for extender.
                            
                            Pass dictionary in a form (no spaces, 
                            backslash before quotes around `value`): 
                                "{\"value\":key}"
                            default = {} (keeps default values as listed below)
                            
                            ''') + \
                            '1. Streamline\n' + textwrap.dedent(extenders.Streamline.__doc__[\
                                    extenders.Streamline.__doc__.find('eta')-4:\
                                        extenders.Streamline.__doc__.find('References')]),
                        default = [{}])
                                                                                                           
    # Plotting at the end
    parser.add_argument('-fp', '--final_plot', action=argparse.BooleanOptionalAction,
                        help= textwrap.dedent('''\
                            Flag indicating to plot the final network.
                            '''))                                                                     

    # parse the arguments from standard input
    args = parser.parse_args()

    if args.input_file is None:
        # Prepare the System from scratch
        
        # Box
        box, branches = Box.construct(initial_condition=args.initial_condition[0], \
                                           **args.kwargs_box[0])
        
        # Network
        network = Network( box=box, branches=branches, active_branches=branches.copy() )
        
        # Trajectory integrator
        if args.trajectory_integrator[0]=='modified_euler':
            trajectory_integrator = trajectory_integrators.modified_euler
        
        # Solver
        if args.pde_solver[0]=='FreeFEM':      
            pde_solver = pde_solvers.FreeFEM(**args.pde_solver_params[0])
        
        # Extender
        if args.extender[0]=='Streamline':
            extender = extenders.Streamline( pde_solver=pde_solver, \
                                            **args.extender_params[0])
        
        # General
        system = System( network=network, extender=extender, \
                        trajectory_integrator=trajectory_integrator, \
                        exp_name=args.output_file[0], \
                        **args.growth_params[0] )
    
    else:
        # Import System from JSON file
        system = System.import_json(input_file=args.input_file[0])
        if args.output_file[0] != '':
            system.exp_name = args.output_file[0]
    
    system.evolve()
    
    if args.final_plot:
        fig, ax = plt.subplots()
        graphics.plot_tree( ax, network=system.network )
        fig.savefig(system.exp_name+'.jpg', bbox_inches='tight')
    
if __name__ == '__main__':
    main()