import matplotlib.pyplot as plt

from reticuler.system import Box, Network, System
from reticuler.extending_kernels import extenders, pde_solvers, trajectory_integrators
from reticuler.user_interface import graphics

# %%
box = Box()
branches = box.construct(INITIAL_CONDITION=0, seeds_x=[0.9, 1.1], initial_lengths=[0.01]*2, height=50)
network = Network( box=box, branches=branches, active_branches=branches.copy() )
extender = extenders.Streamline( pde_solver=pde_solvers.FreeFEM() )
system = System(network=network, extender=extender,
                trajectory_integrator=trajectory_integrators.modified_euler, )
print(system.growth_gauges)

system.extender.DS = 0.05
system.extender.BIFURCATION_TYPE = 1
system.extender.BIFURCATION_THRESH=0.6
system.GROWTH_THRESH_TYPE = 0
system.GROWTH_THRESH = 20

system.FILE_NAME = 'g:\\My drive\\Research\\Network simulations\\reticuler\\tests\\test_dumper.json'
system.DUMP_EVERY = 5
system.evolve()

# %% 
# exp_file = 'g:\\My drive\\Research\\Network simulations\\reticuler\\tests\\test_dumper'
# dumper.export_json(system=system, output_file=exp_file+'.json')

fig, ax = plt.subplots()
graphics.plot_tree( ax, network=system.network)
fig.savefig(system.FILE_NAME+'.pdf', bbox_inches='tight')

# %%
from reticuler import System
from reticuler.user_interface import graphics
import matplotlib.pyplot as plt

exp_file = 'g:\\My drive\\Research\\Network simulations\\reticuler\\tests\\test_dumper'

system = System.import_json(input_file=exp_file+'.json')

fig, ax = plt.subplots()
graphics.plot_tree( ax, network=system.network)
# fig.savefig(exp_file+'2.pdf', bbox_inches='tight')