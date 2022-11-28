import time
import matplotlib.pyplot as plt

from reticuler.system import Box, Network, System
from reticuler.extending_kernels import extenders, pde_solvers, trajectory_integrators
from reticuler.user_interface import graphics

# Import System from JSON file
system = System.import_json(input_file='eta15')
system.exp_name = 'eta15v2'

# fig, ax = plt.subplots()
# graphics.plot_tree(ax, system.network)

system.growth_thresh_type = 0
system.growth_thresh = system.growth_gauges[0] + 2

system.evolve()

# script = system.extender.pde_solver._FreeFEM__prepare_script(system.network)

# start_clock = time.time()
# with open("test2.edp","w") as edp_temp_file:
#     edp_temp_file.write(script)
# print(
#     "\n End of the simulation. Time: {clock:.2f}s".format(
#         clock=time.time() - start_clock
#     )
# )
