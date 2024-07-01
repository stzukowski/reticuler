"""Trimmers to execute backward step.

Classes:
    BackwardModifiedEulerMethod

"""

import numpy as np
import copy


class BackwardModifiedEulerMethod:
    """A class to integrate tip trajectories with reversed modified Euler's method and streamline algorithm.

    Reversed modified Euler's method:

    .. math:: x(n-1) = x(n) - dt \cdot 0.5 \cdot (v[x(n)] + v[x(n-1)]).

    Attributes
    ----------
    pde_solver : PDESolver
    eta : float, default 1.0
        The growth exponent (v=a1**eta).
    ds : float, default 0.01
        A distance over which the fastest branch in the network
        will move in each timestep.    
    max_approximation_step : int, default 3
        Number of approximation steps:
            - 0:  explicit Euler's method
            - 1:  Heuns's method
            - >1: modified Euler's method   
    min_distance : float, default 5e-4
        Minimal distance between two nodes after trimming.
    inflow_thresh : float, default 0.05
        Threshold to put asleep the tips with less than ``inflow_thresh``
        of max flux/velocity.

    References
    ----------
    .. [Ref1] "Through history to growth dynamics: backward evolution of spatial networks",
            S. Żukowski, P. Morawiecki, H. Seybold, P. Szymczak, Sci Rep 12, 20407 (2022). 
            https://doi.org/10.1038/s41598-022-24656-x
    .. [Ref3] https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)

    """
    def __init__(
        self,
        pde_solver,
        eta=0.0,
        ds=0.01,
        max_approximation_step=3,
        inflow_thresh = 0.05,
        min_distance = 5e-4,
    ):
        """Initialize BackwardModifiedEulerMethod.

        Parameters
        ----------
        pde_solver : PDESolver
        eta : float, default 1.0
        ds : float, default 0.01
        max_approximation_step : int, default 3
        inflow_thresh : float, default 0.05
        min_distance : float, default 5e-4

        Returns
        -------
        None.

        """
        self.pde_solver = pde_solver

        self.eta = eta
        self.ds = ds

        self.max_approximation_step = max_approximation_step
        self.min_distance = min_distance  # avoid two nodes very close to each other after trimming
        
        # less than `inflow_thresh` of max flux/velocity puts branches asleep
        self.inflow_thresh = inflow_thresh
        
    def __check_moving_conditions(self, network):
        """Check moving conditions."""
        a1 = self.pde_solver.flux_info[:, 0]
        max_a1 = np.max(a1)
        # (first condition for low eta, second for high)
        are_moving = np.logical_and(a1/max_a1 > self.inflow_thresh,
                                   (a1/max_a1)**self.eta > self.inflow_thresh)
        # shallow copy of active_branches (creates new list instance, but the elements are still the same)
        branches_to_iterate = network.active_branches.copy()
        for i, branch in enumerate(branches_to_iterate):
            if not are_moving[i]:
                network.sleeping_branches.append(branch)
                network.active_branches.remove(branch)
                # print("! Branch {ID} is sleeping !".format(ID=branch.ID))                
        return are_moving

    
    def __explicit_network_trim(self, test_network, drs_0, backward_branches, backward_bifurcations, BEA_step):
        """Trim branches with given ``drs``."""

        dt_factor = 1
        branches_to_iterate = test_network.active_branches.copy()
        dts_to_bif = np.zeros((len(backward_bifurcations.mother_IDs),\
                               len(branches_to_iterate)))
        for i, forward_branch in enumerate(branches_to_iterate):
            backward_branch = backward_branches[forward_branch.ID]
            # check the distances from the tip to bifurcation point
            segment_lengths = np.linalg.norm(
                forward_branch.points[1:]-forward_branch.points[:-1], axis=1)
            length_from_tip = np.cumsum(segment_lengths[::-1])
            ind_bifurcation = backward_bifurcations.mother_IDs==backward_branch.mother_ID
            
            # bifurcation reached
            if drs_0[i]+self.min_distance >= length_from_tip[-1]:
                # what time does it take for tip i to reach the bif
                dts_to_bif[ind_bifurcation, i] = length_from_tip[-1] / drs_0[i]
        
        counts_reached = np.sum(dts_to_bif>0, axis=1) # how many tips reached bifs
        # which bifs were reached for the 2nd time
        are_bifs_reached_second_time = np.logical_or(
                                    counts_reached==2, \
                                    np.logical_and(counts_reached==1, \
                                                   backward_bifurcations.flags==1)
                                        )
        if are_bifs_reached_second_time.any():
            # we change dt, so that the tip which is reaching bif the 2nd
            # stops perfectly at it (np.max)
            # if there are >1 bifs reached, we choose the smallest dt 
            dt_factor = np.min( np.max(dts_to_bif[are_bifs_reached_second_time], axis=1) )
            drs_0 = drs_0 * dt_factor
            
        drs = drs_0.copy()
        # first column: branch disappears (reached bif for the 1st time)
        # second column: branch is replaced with mother branch
        are_living_after_bif = np.ones((len(branches_to_iterate),2), dtype=bool)
        for i, forward_branch in enumerate(branches_to_iterate):
            backward_branch = backward_branches[forward_branch.ID]
            # check the distances from the tip to bifurcation point
            segment_lengths = np.linalg.norm(
                forward_branch.points[1:]-forward_branch.points[:-1], axis=1)
            length_from_tip = np.cumsum(segment_lengths[::-1])
                   
            # bifurcation reached
            if drs[i]+self.min_distance >= length_from_tip[-1]:
                drs[i] = drs[i] - (length_from_tip[-1]) #  - self.min_distance
                backward_branch.add_point(forward_branch.points[0], 0, BEA_step) 
                                
                mask = test_network.branch_connectivity[:,1]!=forward_branch.ID
                test_network.branch_connectivity = test_network.branch_connectivity[mask,...]
                test_network.branches.remove(forward_branch)
                ind_branch = test_network.active_branches.index(forward_branch)
                
                # if we reach the border
                if backward_branch.mother_ID==-1:
                    mask = test_network.box.seeds_connectivity[:,1]!=forward_branch.ID
                    test_network.box.seeds_connectivity = test_network.box.seeds_connectivity[mask,...]
                    test_network.active_branches.pop(ind_branch)
                    are_living_after_bif[i,:] = False
                # bifurcation point
                else:
                    ind_bifurcation = backward_bifurcations.mother_IDs==backward_branch.mother_ID
                    if backward_bifurcations.flags[ind_bifurcation] == 0:
                        print("! Branch {ID} reached bifurcation {bifID} ! (1st one)".format(ID=backward_branch.ID, bifID=backward_branch.mother_ID))
                        backward_bifurcations.flags[ind_bifurcation] = 1
                        are_living_after_bif[i,:] = False
                        # branch is later poped from initial_network (back in the trim method)
                        test_network.active_branches.pop(ind_branch)
                        
                        # total length of the remaining branches connected to the mother branch
                        remaining_length, remaining_IDs = self.__calculate_remaining_length(test_network, backward_branch.mother_ID)
                        inds_remaining = [i for i, b in enumerate(branches_to_iterate) if b.ID in remaining_IDs]
                        
                        # if drs[i]>0 when tip(i) reaches the bifPoint, then dt was too big;
                        # we compansate the effect by adding the 'virtual length'
                        virtual_length = drs[i] / drs_0[i] * np.sum(drs_0[inds_remaining])
                        
                        # if remaining branches weren't moved yet we have to subtract their drs
                        length_mismatch = remaining_length + virtual_length - np.sum(drs[inds_remaining])
                        
                        # if length_mismatch < 0 then we have to reverse the above line and calculate virtual_length for the remaining tip
                        if length_mismatch < 0:
                            virtual_length = (np.sum(drs[inds_remaining])-remaining_length) / np.sum(drs_0[inds_remaining]) * drs_0[i]
                            length_mismatch = virtual_length - drs[i]
                        
                        backward_bifurcations.length_mismatch[ind_bifurcation] = length_mismatch
                        drs[i] = 0
                        
                    elif backward_bifurcations.flags[ind_bifurcation] == 1:
                        print("! Branch {ID} reached bifurcation {bifID} ! (2nd one)".format(ID=backward_branch.ID, bifID=backward_branch.mother_ID))
                        backward_bifurcations.flags[ind_bifurcation] = 2
                        
                        mother_branch = [b for b in test_network.branches if b.ID==backward_branch.mother_ID][0]
                        forward_branch = mother_branch
                        test_network.active_branches[ind_branch] = mother_branch
                        
            # bifurcation not reached
            else:
                # how many points to remove
                to_remove = np.sum(drs[i]+self.min_distance > length_from_tip)
                if to_remove>0:
                    forward_branch.points = forward_branch.points[:-to_remove]
                    forward_branch.steps = forward_branch.steps[:-to_remove]
                    drs[i] = drs[i] - (length_from_tip[to_remove-1]) #  - self.min_distance)
                
                # shifting the last point
                if drs[i]!=0:
                    tip_versor = forward_branch.points[-1] - forward_branch.points[-2]
                    tip_versor = tip_versor / np.linalg.norm(tip_versor)
                    forward_branch.points[-1] = forward_branch.points[-1] - tip_versor * drs[i]
                    backward_branch.add_point(forward_branch.points[-1], len(forward_branch.points)-1, BEA_step) 
                    drs[i] = 0
                
        if (counts_reached==2).any():       
            inds_after_bif = np.empty(len(branches_to_iterate), dtype=int)
            for i, forward_branch in enumerate(branches_to_iterate):
                if forward_branch not in test_network.active_branches:
                    are_living_after_bif[i,0] = True
                    inds_after_bif[i] = test_network.active_branches.index(mother_branch)
                else:
                    inds_after_bif[i] = test_network.active_branches.index(forward_branch)
        else:
            inds_after_bif = np.arange(len(test_network.active_branches))
                        
        return are_living_after_bif, inds_after_bif, dt_factor, drs_0
    
    def __calculate_remaining_length(self, network, mother_ID):
        to_check = network.branch_connectivity[network.branch_connectivity[:,0]==mother_ID,1]
        branch_IDs = np.array([b.ID for b in network.branches])
        remaining_length = 0
        remaining_IDs = []
        while to_check.size:
            next_ID = to_check[0]
            remaining_length = remaining_length + network.branches[np.where(branch_IDs==next_ID)[0][0]].length()
            
            to_add = network.branch_connectivity[network.branch_connectivity[:,0]==next_ID,1]
            if to_add.size:
                to_check = np.concatenate((to_check, to_add))
                remaining_IDs.append(next_ID)
            else:
                remaining_IDs.append(next_ID)
            to_check = to_check[1:]
        return remaining_length, remaining_IDs
    
    def trim(self, network, backward_branches_0, backward_bifurcations_0, BEA_step):
        """Perform backward step with the reversed modified Euler's method and the streamline algorithm.
    
        Parameters
        ----------
        network : Network
            An object of class Network.
        backward_branches_0 : list
            A list of all backward branches.
        backward_bifurcations_0 : BackwardBifurcations
            An object of class BackwardBifurcations.
        BEA_step : int
            Current step of the BEA.            
    
        Returns
        -------
        initial_network : Network
            The initial network with updated active_branches list (removed branches: (i) aren't moving, (ii) reached bifurcation)
        test_network : Network
            A network after backward evolution step.
        backward_branches : list
            An updated list of all backward branches.
        backward_bifurcations : BackwardBifurcations
            An updated object of class BackwardBifurcations.
        dt : float
            Time of (backward) growth at the current BEA step.
        
        Returns
        -------
        None.
                
        """
        # activate sleeping branches        
        network.active_branches = network.sleeping_branches + network.active_branches
        network.sleeping_branches = []
        
        # v[x(n)]: finding velocity at the starting point
        self.pde_solver.solve_PDE(network)
        velocity_0 = self.pde_solver.flux_info[:, 0]**self.eta
        dt_0 = self.ds / np.max(velocity_0)
        drs_0 = dt_0 * velocity_0

        # are moving?
        are_moving = self.__check_moving_conditions(network) # here we put branches to sleep
        drs_0 = drs_0[are_moving]
        initial_network = network.copy()
        
        # x(n-1): trimming drs_0 from test_network
        backward_branches = copy.deepcopy(backward_branches_0)
        backward_bifurcations = copy.deepcopy(backward_bifurcations_0)
        test_network = network.copy()
        are_living_after_bif, inds_after_bif, dt_factor, drs_0 = \
            self.__explicit_network_trim(test_network, drs_0, backward_branches, \
                                         backward_bifurcations, BEA_step)
        print('dt_factor:', dt_factor)
        print(inds_after_bif)
        dt_0 = dt_0 * dt_factor 
        
        drs = drs_0.copy()
        print(drs)
        dt = dt_0
        approximation_step = 0
        # APPROXIMATION LOOP
        while test_network.active_branches and \
                    approximation_step < self.max_approximation_step and not \
                        (~are_living_after_bif[:,0]).all():
            approximation_step = approximation_step + 1

            # v[x(n-1)]: finding velocity at the x(n-1) point
            self.pde_solver.solve_PDE(test_network)
            velocity_1 = self.pde_solver.flux_info[:, 0]**self.eta
            drs_1 = dt_0 * velocity_1

            # 0.5 * dt * (v[x(n)] + v[x(n-1)]): average drs
            drs[are_living_after_bif[:,0]] = (drs_0[are_living_after_bif[:,0]] + drs_1[inds_after_bif]) / 2
            velocity = drs / dt_0
            dt = self.ds / np.max(velocity)
            drs = dt * velocity
            print(drs)

            # improved x(n-1)
            backward_branches = copy.deepcopy(backward_branches_0)
            backward_bifurcations = copy.deepcopy(backward_bifurcations_0)
            test_network = network.copy()
            are_living_after_bif, inds_after_bif, dt_factor, drs = \
                self.__explicit_network_trim(test_network, drs, backward_branches, \
                                             backward_bifurcations, BEA_step)
            print('dt_factor:', dt_factor)
            dt = dt * dt_factor
            print(inds_after_bif)
        
        initial_network.active_branches = [b for i, b in enumerate(network.active_branches) if are_living_after_bif[i,1]]
        
        return initial_network, test_network, backward_branches, backward_bifurcations, dt        
        
