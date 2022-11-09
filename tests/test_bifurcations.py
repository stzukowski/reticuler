import numpy as np
import matplotlib.pyplot as plt
import time

from reticulator.system import Box, Network, System
from reticulator.extending_kernels import Extenders, PDESolvers, TrajectoryIntegrators
from reticulator.dumpers import Graphics, Dumper
from reticulator.others import Tests

# %%
box = Box()
branches = box.construct(INITIAL_CONDITION=0, seeds_x=[1.5], initial_lengths=[0.01], height=50)
network = Network( box=box, branches=branches, active_branches=branches.copy() )
extender = Extenders.Streamline( solver=PDESolvers.FreeFEM(),
                                BIFURCATION_TYPE = 1)
system = System(exp_file='g:\\My drive\\Research\\Network simulations\\Reticulator\\test_bifurcations',
                network=network, extender=extender,
                trajectory_integrator=TrajectoryIntegrators.modified_euler)

system.extender.DS = 0.01
system.GROWTH_THRESH_TYPE = 1
system.GROWTH_THRESH = 5
# %%
system.reticulate()
# %%
fig, ax = plt.subplots()
Graphics.plot_tree(ax, network, height=5.5, marker='.', ms=3)
    
# %% 
dumper = Dumper()
dumper.export_json(system)