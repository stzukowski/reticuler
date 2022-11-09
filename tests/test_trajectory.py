import matplotlib.pyplot as plt
import time

from reticulator.system import Box, Network, System
from reticulator.extending_kernels import Extenders, PDESolvers, TrajectoryIntegrators
from reticulator.dumpers import graphics, dumper

# %%
box = Box()
branches = box.construct(INITIAL_CONDITION=0, seeds_x=[1.5], initial_lengths=[0.01], height=50)
network = Network( box=box, branches=branches )
extender = Extenders.Streamline( solver=PDESolvers.FreeFEM() )
system = System(exp_file='g:\\My drive\\Research\\Network simulations\\Reticulator\\test_trajectory',
                network=network, extender=extender,
                trajectory_integrator=TrajectoryIntegrators.modified_euler)

extender.DS = 0.05
system.GROWTH_THRESH_TYPE = 1
system.GROWTH_THRESH = 5
system.reticulate()
# %%
# Comparison with analytical results on Y slices (<\Delta x (y)>, as in 
# "Path selection in the growth of rivers", Cohen et al., PNAS 2015)

# Constructing analytical trajectory on the same Y slices
# (time consuming!)

# start_clock = time.time()
# mean_error, trajectory_analytical = Tests.construct_and_compare_with_analytical(system.network.branches[0].points)
# print('Mean error: ', mean_error)
# print('Comparison time: {clock:.2f}s'.format(clock=time.time()-start_clock))
# %%
# Comparing with already generated analytical trajectory
start_clock = time.time()
mean_error = Tests.compare_with_analytical(system.network.branches[0].points)
trajectory_analytical = Tests.analytical_trajectory_height5()
print('Mean error: ', mean_error)
print('Comparison time: {clock:.2f}s'.format(clock=time.time()-start_clock))
# %%
fig, ax = plt.subplots()
Graphics.plot_tree(ax, network, height=system.GROWTH_THRESH, linestyle='None', marker='.', ms=7)
# ax.plot(*trajectory_analytical.T, 'r.-', ms=7, lw=1)