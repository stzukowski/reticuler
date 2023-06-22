"""Extenders to find the next position of the tip.

Classes:
    Streamline

"""

import numpy as np


class Streamline:
    """Extender based on the streamline algorithm [Ref1]_.

    Attributes
    ----------
    pde_solver : PDESolver
    eta : float, default 1.0
        The growth exponent (v=a1**eta).
        High values increase competition between the branches.
        Low values stabilize the growth.
    ds : float, default 0.01
        A distance over which the fastest branch in the network
        will move in each timestep.
    bifurcation_type : int, default 0
        - 0: no bifurcations
        - 1: a1 bifurcations (velocity criterion)
        - 2: a3/a1 bifurcations (bimodality criterion)
        - 3: random bifurcations
    bifurcation_thresh : float, default 0
        Threshold for the bifurcation criterion.
    bifurcation_angle : float, default 2pi/5
        Angle between the daughter branches after bifurcation.
        Default angle (72 degrees) corresponds to the analytical solution
        for fingers in a diffusive field.
    inflow_thresh : float, default 0.05
        Threshold to put asleep the tips with less than ``inflow_thresh``
        of max flux/velocity.
    distance_from_bif_thresh : float, default 2.1*``ds``
        A minimal distance the tip has to move from the previous bifurcations
        to split again.

    References
    ----------
    .. [Ref1] "Through history to growth dynamics: backward evolution of spatial networks",
            S. Żukowski, P. Morawiecki, H. Seybold, P. Szymczak, Sci Rep 12, 20407 (2022). 
            https://doi.org/10.1038/s41598-022-24656-x

    """

    def __init__(
        self,
        pde_solver,
        eta=1.0,
        ds=0.01,
        bifurcation_type=0,
        bifurcation_thresh=0,
        bifurcation_angle=2 * np.pi / 5,
        inflow_thresh=0.05
    ):
        """Initialize Streamline.

        Parameters
        ----------
        pde_solver : object of class pde_solvers
        eta : float, default 1.0
        ds : float, default 0.01
        bifurcation_type : int, default 0
        bifurcation_thresh : float, default 0
        bifurcation_angle : float, default 2pi/5

        Returns
        -------
        None.

        """
        self.pde_solver = pde_solver

        self.eta = eta
        self.ds = ds

        self.bifurcation_type = bifurcation_type  # no bifurcations, a1, a3/a1, random
        self.bifurcation_thresh = bifurcation_thresh
        if not bifurcation_thresh:
            if self.bifurcation_type == 1:
                self.bifurcation_thresh = 0.8  # a1 bifurcations
            if self.bifurcation_type == 2:
                self.bifurcation_thresh = -0.1  # a3/a1 bifurcations
            if self.bifurcation_type == 3:
                self.bifurcation_thresh = 3 * ds
                # random bifurcations: bif_probability
        self.bifurcation_angle = bifurcation_angle  # 2*np.pi/5

        # less than `inflow_thresh` of max flux/velocity puts branches asleep
        self.inflow_thresh = inflow_thresh
        self.distance_from_bif_thresh = 2.1 * ds

    def __rotation_matrix(self, angle):
        """Construct a matrix to rotate a vector by an ``angle``.

        Parameters
        ----------
        angle : float

        Returns
        -------
        array
            An 2-2 array.

        Examples
        --------
        >>> rot = self.__rotation_matrix(self, angle)
        >>> rotated_vector = np.dot(rot, vector)

        """
        return np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )

    def check_bifurcation_and_moving_conditions(self, network):
        """Check bifurcation and moving conditions."""

        a1 = self.pde_solver.a1a2a3_coefficients[..., 0]
        max_a1 = np.max(a1)
        # checking which branches are_moving
        # (first condition for low eta, second for high)
        are_moving = np.logical_and(a1/max_a1 > self.inflow_thresh,
                                    (a1/max_a1)**self.eta > self.inflow_thresh)

        # shallow copy of active_branches (creates new list instance, but the elements are still the same)
        branches_to_iterate = network.active_branches.copy()
        for i, branch in enumerate(branches_to_iterate):
            a1 = self.pde_solver.a1a2a3_coefficients[i, 0]
            a3 = self.pde_solver.a1a2a3_coefficients[i, 2]
            if (
                self.bifurcation_type
                and branch.length() > self.distance_from_bif_thresh
            ):
                # the second condition above is used to avoid many bifurcations
                # in almost one point which can occur while ds is very small
                if (self.bifurcation_type == 1 and a1 > self.bifurcation_thresh) or (
                    self.bifurcation_type == 2 and a3 / a1 < self.bifurcation_thresh
                ):
                    branch.is_bifurcating = True
                elif self.bifurcation_type == 3:
                    p = self.bifurcation_thresh * (a1 / max_a1) ** self.eta
                    r = np.random.uniform(0, 1)  # uniform distribution [0,1)
                    if p > r:
                        branch.is_bifurcating = True
                        
            if not are_moving[i]:
                network.sleeping_branches.append(branch)
                network.active_branches.remove(branch)
                print("! Branch {ID} is sleeping !".format(ID=branch.ID))

        return are_moving

    def __streamline_extension(self, beta, dr):
        """Calculate a vector over which the tip is shifted.

        Derived from the fact that the finger proceeds along a unique
        streamling going through the tip.

        Parameters
        ----------
        beta : float
            a1/a2 value
        dr : float
            A distance over which the tip is moving.

        Returns
        -------
        dR : array
            An 1-2 array.

        """
        if np.abs(beta) < 1000:
            y = ((beta**2) / 9) * ((27 * dr / (2 * beta**2) + 1) ** (2 / 3) - 1)
        else:
            y = dr - (9*dr**2)/(4*beta**2) + (27*dr**3) / \
                (2*beta**4) - (1701*dr**4)/(16*beta**6)
        x = np.around(
            np.sign(beta) * 2 * ((y**3 / beta**2) +
                                 (y / beta) ** 4) ** (1 / 2), 9)                                                      
        return np.array([x, y])

    def find_test_dRs(self, network, is_BEA_off):
        """Find a single test shift over which the tip is moving.

        Parameters
        ----------
        network : object of class Network

        Returns
        -------
        dRs_test : array
            An n-2 array with dx and dy shifts for each tip.

        """
        # running PDE solver
        # self.pde_solver.a1a2a3_coefficients are updated in FreeFEM solver
        self.pde_solver.solve_PDE(network)

        if is_BEA_off:
            # normalize dr, so that the fastest tip moves over ds
            dr_norm = np.max(self.pde_solver.a1a2a3_coefficients[..., 0] ** self.eta)
        else:
            # dr_norm = 1, ds <=> dt
            dr_norm = 1
        dRs_test = np.empty((len(network.active_branches), 2))
        for i, branch in enumerate(network.active_branches):
            a1 = self.pde_solver.a1a2a3_coefficients[i, 0]
            a2 = self.pde_solver.a1a2a3_coefficients[i, 1]
            beta = a1 / a2

            # __streamline_extension formula is derived in the coordinate
            # system where the tip segment lies on a negative Y axis;
            # hence, we rotate obtained dR vector to that system
            tip_angle = np.pi / 2 - branch.tip_angle()
            dr = self.ds * a1**self.eta / dr_norm
            dRs_test[i] = np.dot(
                self.__rotation_matrix(
                    tip_angle), self.__streamline_extension(beta, dr)
            )
        return dRs_test

    def assign_dRs(self, network, dRs):
        """Assign ``dRs`` to each branch in ``network``."""
        for i, branch in enumerate(network.active_branches):
            if branch.is_bifurcating:
                branch.dR = [
                    np.dot(
                        self.__rotation_matrix(-self.bifurcation_angle / 2), dRs[i]),
                    np.dot(self.__rotation_matrix(
                        self.bifurcation_angle / 2), dRs[i]),
                ]
            else:
                branch.dR = dRs[i]


# class Royal:
#     def __init__(self, solver)

# class BEM:
#     def __init__(self, geometry):

# class BARW:
#     def __init__(self, geometry):
