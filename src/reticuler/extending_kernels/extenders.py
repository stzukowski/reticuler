"""Classes to integrate tip trajectories.

Classes:
    ModifiedEulerMethod_Streamline

"""

import numpy as np


class ModifiedEulerMethod:
    """A class to integrate tip trajectories with modified Euler's method.
    
    Modified Euler's method [Ref3]_ (x(n+1) needs to be found implicitly):

    .. math:: x(n+1) = x(n) + dt \cdot 0.5 \cdot (v[x(n)] + v[x(n+1)]).
    
    Attributes
    ----------
    pde_solver : pde_solver
    is_reconnecting : bool, default False
        A boolean condition if potential reconnections should be checked after each time step.
    max_approximation_step : int, default 3
        Number of approximation steps:
            - 0:  explicit Euler's method
            - 1:  Heuns's method
            - >1: modified Euler's method    
            
    References
    ----------
    .. [Ref3] https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)
    
    """
    
    def __init__(
        self,
        pde_solver,
        is_reconnecting=False,
        max_approximation_step=3
    ):
        """Initialize ModifiedEulerMethod_Streamline.

        Parameters
        ----------
        pde_solver : object of class pde_solvers
        is_reconnecting : bool, default False
        max_approximation_step : int, default 3

        Returns
        -------
        None.

        """
        self.pde_solver = pde_solver
        self.is_reconnecting = is_reconnecting
        # max_approximation_step for the trajectory integration
        self.max_approximation_step = max_approximation_step
     
    def integrate(self, network, step, is_dr_normalized=True):
        """Integrate tip trajectories with modified Euler's method.
    
        Parameters
        ----------
        network : Network
            An object of class Network.
        is_dr_normalized : bool, default True
            A boolean condition if the Backward Evolution Algorithm is off.
            If False `dt`=`self.ds`.
    
        Returns
        -------
        dt : float
            Time of growth at the current evolution step.
        flux_info_0 : array
            An array of a1a2a3 coefficients for each tip in the network.
    
        """
        # running PDE solver
        # self.pde_solver.flux_info are updated in FreeFEM solver
        out_solver = self.pde_solver.solve_PDE(network) # returns flux_info_0 for backward and rim_xy_fluxes for leaf
            
        # x[n + 1] = x[n] + dt * v[x(n)]: finding position n+1 with explicit Euler
        dRs_0, dt_0 = self.pde_solver.find_test_dRs(network, is_dr_normalized, is_zero_approx_step=True)
        dt = dt_0
        
        # moving network tips
        network.move_tips(self, step=step)
        network0 = network.copy()
    
        did_reconnect = False
        if self.is_reconnecting:
            did_reconnect = network.reconnect(self.pde_solver, step)
            
        if did_reconnect:
            print("Reconnected branch, skipping Modified Euler Method steps.")
        else:
            dRs_test = dRs_0.copy()
            approximation_step = 0
            # APPROXIMATION LOOP - we end 'max_approximation_step' steps
            while approximation_step < self.max_approximation_step:
                approximation_step = approximation_step + 1
        
                self.pde_solver.solve_PDE(network)
                
                # v[ x(n+1)] ]: finding velocity at the next point
                dRs_1, _ = self.pde_solver.find_test_dRs(network, is_dr_normalized)
                
                # average dR
                dRs_test = (dRs_0 + dRs_1) / 2
                if is_dr_normalized:
                    dRs_test = self.pde_solver.ds * dRs_test / np.max(  np.linalg.norm(dRs_test, axis=1) )
                    
                    # normally division dX/a1^eta would give single dt
                    # due to the modified Euler's algorithm (dR = (dRs_0+dRs_1)/2 )
                    # dt is not perfectly the same for different tips, so we take a mean
                    dt = np.mean( np.linalg.norm(dRs_test, axis=1) / \
                                 self.pde_solver.flux_info[...,0]**self.pde_solver.eta )
                    
                # adjust tip positions in test_network
                for i, branch in enumerate(network.active_branches):
                    branch.points[-1] = network0.active_branches[i].points[-1] - dRs_0[i] + dRs_test[i]
                
                # print('Forth loop, approximation step: {step}.'.format(step=approximation_step) )
                # print('dRs: ', dRs_test)    
        
        return [dt, out_solver]
