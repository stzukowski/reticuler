"""Classes to integrate tip trajectories.

Classes:
    ModifiedEulerMethod_Streamline

"""

import logging
import numpy as np

from reticuler.utilities.misc import NEUMANN_0, DIRICHLET_0, DIRICHLET_1

logger = logging.getLogger("reticuler")

class ModifiedEulerMethod:
    r"""A class to integrate tip trajectories with modified Euler's method.
    
    Modified Euler's method [Ref3]_ (x(n+1) needs to be found implicitly):

    .. math:: x(n+1) = x(n) + dt \cdot 0.5 \cdot (v[x(n)] + v[x(n+1)]).
    
    Attributes
    ----------
    pde_solver : pde_solver
    is_reconnecting : bool, default False
        A boolean condition if potential reconnections should be checked after each time step.
    max_approximation_step : int, default 1
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
        max_approximation_step=1
    ):
        """Initialize ModifiedEulerMethod_Streamline.

        Parameters
        ----------
        pde_solver : object of class pde_solvers
        is_reconnecting : bool, default False
        max_approximation_step : int, default 1

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
            If False ``dt``=``self.ds``.
    
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
        for branch, dR in zip(network.active_branches, dRs_0):
            branch.extend(step, dR)
        network1 = network.copy()
    
        did_reconnect = False
        if self.is_reconnecting:
            did_reconnect = network.reconnect(self.pde_solver, step)
            if did_reconnect:
                logger.info("Reconnected branch, skipping Modified Euler Method steps.")

        dRs_test = dRs_0.copy()
        approximation_step = 0
        # APPROXIMATION LOOP - we end 'max_approximation_step' steps
        while approximation_step < self.max_approximation_step and not did_reconnect:
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
                branch.points[-1] = network1.active_branches[i].points[-1] - dRs_0[i] + dRs_test[i]
            
            # print('Forth loop, approximation step: {step}.'.format(step=approximation_step) )
            # print('dRs: ', dRs_test) 
            
            if self.is_reconnecting:
                did_reconnect = network.reconnect(self.pde_solver, step)
                if did_reconnect:
                    logger.info("Reconnected branch, (possibly) skipping Modified Euler Method steps.")
                    break
        
        return [dt, out_solver]


class ModifiedEulerMethod_Boundary(ModifiedEulerMethod):
    r"""A class to integrate tip/boundary trajectories with modified Euler's method.
    
    Attributes
    ----------
    pde_solver, is_reconnecting, max_approximation_step inherited from ModifiedEulerMethod
    
    """
    
    def __init__(
        self,
        pde_solver,
        is_reconnecting=False,
        max_approximation_step=1,
    ):
        """Initialize ModifiedEulerMethod_Streamline.

        Parameters
        ----------
        pde_solver : object of class pde_solvers
        is_reconnecting : bool, default False
        max_approximation_step : int, default 1

        Returns
        -------
        None.

        """
        super().__init__(pde_solver, is_reconnecting, max_approximation_step)

    def integrate(self, network, step, is_dr_normalized=True):
        """Integrate tip trajectories with modified Euler's method.
    
        Parameters
        ----------
        network : Network
            An object of class Network.
        is_dr_normalized : bool, default True
            A boolean condition if the Backward Evolution Algorithm is off.
            If False ``dt``=``self.ds``.
    
        Returns
        -------
        dt : float
            Time of growth at the current evolution step.
        flux_info_0 : array
            An array of a1a2a3 coefficients for each tip in the network.
    
        """
        # running PDE solver
        # self.pde_solver.flux_info are updated in FreeFEM solver
        out_solver_0 = self.pde_solver.solve_PDE(network)
            
        # x[n + 1] = x[n] + dt * v[x(n)]: finding position n+1 with explicit Euler
        dRs_0, dt_0 = self.pde_solver.find_test_dRs(network, is_dr_normalized, is_zero_approx_step=True)
        dRs_boundary_0 = self.pde_solver.find_test_dRs_boundary(network, dt_0)
        dt = dt_0
        
        # moving network tips and boundary
        for branch, dR in zip(network.active_branches, dRs_0):
            branch.extend(step, dR)
        mask_moving_boundary = network.box.boundary_conditions==DIRICHLET_1; 
        mask_moving_boundary[1:] |= mask_moving_boundary[:-1].copy() # "spill" True on point i+1 (i-last detected)
        network.box.points[mask_moving_boundary] += dRs_boundary_0
        network1 = network.copy()
    
        did_reconnect = False
        if self.is_reconnecting:
            did_reconnect = network.reconnect(self.pde_solver, step)
            
        if did_reconnect:
            logger.info("Reconnected branch, skipping Modified Euler Method steps.")
        else:
            dRs_test = dRs_0.copy()
            dRs_boundary_test = dRs_boundary_0.copy()
            approximation_step = 0
            # APPROXIMATION LOOP - we end at 'max_approximation_step' steps
            # import matplotlib.pyplot as plt
            # plt.plot(np.linalg.norm(dRs_boundary_0, axis=1),'.-', ms=5); plt.show;
            # plt.plot(self.pde_solver.flux_info_boundary); plt.show;
            while approximation_step < self.max_approximation_step:
                approximation_step = approximation_step + 1
        
                self.pde_solver.solve_PDE(network)
                
                # v[ x(n+1)] ]: finding velocity at the next point
                dRs_1, dt_1 = self.pde_solver.find_test_dRs(network, is_dr_normalized)
                
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
                    branch.points[-1] = network1.active_branches[i].points[-1] - dRs_0[i] + dRs_test[i]
                
                # # MOVING BOUNDARY
                dRs_boundary_1 = self.pde_solver.find_test_dRs_boundary(network, dt_1)
                alpha = 0.5 # Typical ModifiedEulerMethod; if alpha=0 - Euler method
                dRs_boundary_test = (1-alpha)*dRs_boundary_0 + alpha*dRs_boundary_1

                # Weighted average (cancels noise better)
                # from scipy.ndimage import uniform_filter1d
                # def noise_amplitude(signal, smooth_window=15):
                #     smoothed = uniform_filter1d(signal, smooth_window, mode='nearest')
                #     return np.std(signal - smoothed)
                # sigma_0 = noise_amplitude(np.linalg.norm(dRs_boundary_0, axis=1))
                # sigma_1 = noise_amplitude(np.linalg.norm(dRs_boundary_1, axis=1))
                # w0 = 1.0 / sigma_0
                # w1 = 1.0 / sigma_1
                # dRs_boundary_test = (w0 * dRs_boundary_0 + w1 * dRs_boundary_1) / (w0 + w1)
                
                # plt.plot(np.linalg.norm(dRs_boundary_1, axis=1),'.-', ms=5); plt.show;
                # plt.plot(np.linalg.norm(dRs_boundary_test, axis=1),'.-', ms=5); plt.show;
                # plt.plot(self.pde_solver.flux_info_boundary); plt.show;
                
                network.box.points[mask_moving_boundary] = network1.box.points[mask_moving_boundary] - dRs_boundary_0 + dRs_boundary_test
                # print('Forth loop, approximation step: {step}.'.format(step=approximation_step) )
                # print('dRs: ', dRs_test)    
        
        self.pde_solver.control_point_density(network, network.box.points[mask_moving_boundary])
        self.pde_solver.box_history.append(network.box.copy())

        return [dt, out_solver_0]