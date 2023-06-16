import sys,os
sys.path.append(os.getcwd())

from reticuler_test.system import Box, Network, System
from reticuler_test.extending_kernels import extenders, pde_solvers, trajectory_integrators
from reticuler_test.user_interface import graphics

# Prepare the System from scratch
# Box
box, branches = Box.construct(initial_condition=1, height=2, seeds_x=[1.5])

# Network
network = Network(box=box, branches=branches, active_branches=branches.copy())

# Trajectory integrator
trajectory_integrator = trajectory_integrators.ModifiedEulerMethod()

# Solver
pde_solver = pde_solvers.FreeFEM(equation=1)

# Extender
extender = extenders.Streamline(pde_solver=pde_solver, ds=0.01, bifurcation_type=2)

# General
system = System(
    network=network,
    extender=extender,
    trajectory_integrator=trajectory_integrator,
    exp_name='eta30',
    growth_thresh_type=2,
    growth_thresh=13,
)
system.evolve()