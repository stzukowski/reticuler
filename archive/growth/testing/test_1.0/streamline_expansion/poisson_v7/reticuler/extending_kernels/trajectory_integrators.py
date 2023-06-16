"""Functions to integrate the tips trajectory.

Functions:
    modified_euler(extender, network, max_approximation_step=3)

"""

import numpy as np


def modified_euler(extender, network, max_approximation_step=3):
    """Integrate tips trajectory with modified Euler's method.

    Modified Euler's method [Ref3]_ (x(n+1) needs to be found implicitly):

    .. math:: x(n+1) = x(n) + dt \cdot 0.5 \cdot (v[x(n)] + v[x(n+1)]).

    Parameters
    ----------
    max_approximation_step : int, default 3
        Number of approximation steps:
            - 0:  explicit Euler's method
            - 1:  Heuns's method
            - >1: modified Euler's method

    Returns
    -------
    None.

    References
    ----------
    .. [Ref3] https://en.wikipedia.org/wiki/Trapezoidal_rule_(differential_equations)

    """
    # x[n + 1] = x[n] + dt * v[x(n)]: finding position n+1 with explicit Euler
    dRs_0 = extender.find_test_dRs(network)
    print('dRs_0: ', dRs_0)
    # checking if each branch isMoving or isBifurcating
    extender.check_bifurcation_and_moving_conditions(network)
    # print('a1, a2, a3:\n',extender.pde_solver.a1a2a3_coefficients)

    # moving test_system by dRs_0
    test_network = network.copy()
    test_network.move_test_tips(dRs_0)

    dRs_test = 0
    approximation_step = 0
    epsilon_thresh = (
        np.sum(np.linalg.norm(dRs_0, axis=1)) / 1000
    )  # probably to optimize
    epsilon = epsilon_thresh + 1
    # APPROXIMATION LOOP - we end when approximating isn't getting any
    # better (epsilon is the measure) or after 'max_approximation_step' steps
    while epsilon > epsilon_thresh and approximation_step < max_approximation_step:
        approximation_step = approximation_step + 1

        # v[ x(n+1)] ]: finding velocity at the next point
        dRs_1 = extender.find_test_dRs(test_network)
        print('dRs_1: ', dRs_1)
        # calculating approximation error and average dR
        epsilon = np.sum(np.linalg.norm(dRs_test - (dRs_0 + dRs_1) / 2, axis=1)) / len(
            dRs_0
        )
        dRs_test = (dRs_0 + dRs_1) / 2
        dRs_test = dRs_test * extender.ds / np.max(np.linalg.norm(dRs_test, axis=1))
        print('dRs_test: ', dRs_test)

        # moving test_system by dR
        test_network = network.copy()
        test_network.move_test_tips(dRs_test)

        # print('Forth loop, approximation step: {step}, epsilon = {eps}'.format(step=approximation_step, eps=epsilon) )

    # normally division dX/a1^eta would give single dt
    # due to the modified Euler's algorithm (dR = (dRs_0+dRs_1)/2 )
    # dt is not perfectly the same for different tips, so we take a mean
    # dt = np.mean( np.linalg.norm(dRs_test, axis=1) / extender.pde_solver.a1a2a3_coefficients[...,0]**self.eta)

    # moving tips with final dR after bifurcations
    extender.assign_dRs(dRs_test, network)
