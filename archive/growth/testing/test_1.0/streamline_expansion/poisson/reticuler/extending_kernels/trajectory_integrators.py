"""Classes to integrate tip trajectories.

Classes:
    ModifiedEulerMethod

"""

import numpy as np


class ModifiedEulerMethod:
    """A class to integrate tip trajectories with modified Euler's method.
    
    Modified Euler's method [Ref3]_ (x(n+1) needs to be found implicitly):

    .. math:: x(n+1) = x(n) + dt \cdot 0.5 \cdot (v[x(n)] + v[x(n+1)]).
    
    Attributes
    ----------
    max_approximation_step : int, default 3
        Number of approximation steps:
            - 0:  explicit Euler's method
            - 1:  Heuns's method
            - >1: modified Euler's method    
            
    References
    ----------
    .. [Ref3] https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)
    
    """
    
    def __init__(self, max_approximation_step=3):
        """Initialize ModifiedEulerMethod.
        
        Parameters
        ----------
        max_approximation_step : int, default 3
        
        Returns
        -------
        None.
        
        """
        self.max_approximation_step = max_approximation_step
        
    def move_test_tips(self, network, dRs):
        """Move test tips (no bifurcations or killing).

        Assign ``dRs`` to each branch in ``network.active_branches`` and extend them.

        """
        for i, branch in enumerate(network.active_branches):
            branch.dR = dRs[i]
            branch.extend()

    def integrate(self, extender, network, is_BEA_off=True):
        """Integrate tip trajectories with modified Euler's method.
    
        Parameters
        ----------
        extender : Extender
            An object of one of the classes from reticuler.extending_kernels.extenders.
        network : Network
            An object of class Network.
        is_BEA_off : bool, default True
            A boolean condition if the Backward Evolution Algorithm is off.
            If True ``a1a2a3_coefficients`` (from the tip position before extension)
            are not returned.
    
        Returns
        -------
        dt : float
            Time of growth at the current evolution step.
        a1a2a3_coefficients_0 : array
            An array of a_i coefficients for each tip in the network.
    
        """
        # x[n + 1] = x[n] + dt * v[x(n)]: finding position n+1 with explicit Euler
        dRs_0 = extender.find_test_dRs(network, is_BEA_off)
        
        # in the BEA we want to move over dt as in the backward step,
        # so no normalization
        a1a2a3_coefficients_0 = extender.pde_solver.a1a2a3_coefficients
            
        # checking if each branch is_moving or is_bifurcating
        are_moving = extender.check_bifurcation_and_moving_conditions(network)
        dRs_0 = dRs_0[are_moving]
        # print('a1, a2, a3:\n',extender.pde_solver.a1a2a3_coefficients)
    
        # moving test_system by dRs_0
        test_network = network.copy()
        self.move_test_tips(test_network, dRs_0)
    
        dRs_test = 0
        approximation_step = 0
        epsilon_thresh = (
            np.sum(np.linalg.norm(dRs_0, axis=1)) / 1000
        )  # probably to optimize
        epsilon = epsilon_thresh + 1
        # APPROXIMATION LOOP - we end when approximating isn't getting any
        # better (epsilon is the measure) or after 'max_approximation_step' steps
        while epsilon > epsilon_thresh and approximation_step < self.max_approximation_step:
            approximation_step = approximation_step + 1
    
            # v[ x(n+1)] ]: finding velocity at the next point
            dRs_1 = extender.find_test_dRs(test_network, is_BEA_off)
            
            # calculating approximation error and average dR
            epsilon = np.sum(np.linalg.norm(dRs_test - \
                            (dRs_0 + dRs_1) / 2, axis=1)) / len(dRs_0)
            dRs_test = (dRs_0 + dRs_1) / 2
            if is_BEA_off:
                dRs_test = dRs_test * extender.ds / np.max(np.linalg.norm(dRs_test, axis=1))
                
            # moving test_system by dR
            test_network = network.copy()
            self.move_test_tips(test_network, dRs_test)
            
            # print('Forth loop, approximation step: {step}, epsilon = {eps}'.format(step=approximation_step, eps=epsilon) )
        if is_BEA_off:
            # normally division dX/a1^eta would give single dt
            # due to the modified Euler's algorithm (dR = (dRs_0+dRs_1)/2 )
            # dt is not perfectly the same for different tips, so we take a mean
            dt = np.mean( np.linalg.norm(dRs_test, axis=1) / extender.pde_solver.a1a2a3_coefficients[...,0]**extender.eta)
        else: 
            dt = extender.ds
        
        extender.assign_dRs(network, dRs_test)
        
        return dt, a1a2a3_coefficients_0
