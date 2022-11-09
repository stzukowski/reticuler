from reticulator.system import Box, Network, System
from reticulator.evolvers import Extenders, PDESolvers
from reticulator.dumper import Graphics

box = Box()
branches = box.construct(INITIAL_CONDITION=0, seeds_x=[0.5], initial_lengths=[0.01], height=50)

print('--- Box ---\n')
print('Points: \n', box.points)
print('\nConnections and boundary conditions:')
print(box.connections_bc())
print('Seeds connectivity: \n', box.seeds_connectivity)

print('\n--- Points ---')
for branch in branches:
    branch.extend_by_dR( dR=[0, 0.01] )
    print('Branch ID: ', branch.BRANCH_ID)
    print('Length: ', branch.length())
    print('Points:\n', branch.points)
    
network = Network( branches=branches )
extender = Extenders.Streamline( solver=PDESolvers.FreeFEM() )
system = System( box=box, network=network, extender=extender, EXP_FILE='g:\\My drive\\Research\\Network simulations\\Reticulator\\test')