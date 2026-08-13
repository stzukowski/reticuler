"""Building blocks of the System.

Classes:
    Box
    Branch
    Network
"""

import logging
import numpy as np

from reticuler.utilities.misc import DIRICHLET_0

logger = logging.getLogger("reticuler")

class Branch:
    """A class of a single branch in a network.

    Attributes
    ----------
    ID : int
        Branch ID.
    points : array
        A 2-n array with xy coordinates of the points composing the branch.
        Chronological order (tip is the last point).
    steps : array
        A 1-n array with evolution steps at which corresponding points were added.
    BC : int or array, default DIRICHLET_0
        The boundary condition on fingers, when solving the equations for the field.
        1. FreeFEM_ThinFingers:
            - DIRICHLET_0 (u=0)
            - DIRICHLET_1 (u=1)
        2. FreeFEM_ThickFingers (diffusive case)
            Mobility ratio between the inside and outside of the fingers.
            mobility_outside = 1, mobilty_inside = branch.BC
        2. FreeFEM_ThickFingers_Elasticity (elastic case)
            Young's modulus and Poisson's ratio of the canals.
    is_bifurcating : bool, default False
        A boolean condition if branch is bifurcating or not.
        (Based on the bifurcation_type and bifurcation_thresh from extender.)

    """

    def __init__(self, ID, points, steps, BC=DIRICHLET_0):
        """Initialize Branch.

        Parameters
        ----------
        ID : int
        points : array
        steps : array
        BC : int or array

        Returns
        -------
        None.

        """
        self.ID = ID

        self.points = points  # in order of creation
        self.steps = steps  # at which step of the evolution the point was added
        self.BC = BC # boundary condition

    def extend(self, step, dR):
        """Add a new point to ``self.points`` (progressed tip) and update ``self.steps``."""
        if np.linalg.norm(dR) < 9e-5:
            logger.warning("! Extremely small dR, tip %d not extended but shifted !", self.ID)
            tip_versor = self.points[-1] - self.points[-2]
            tip_versor = tip_versor / np.linalg.norm(tip_versor)
            self.points[-1] = self.points[-1] + tip_versor * dR
        else:
            self.points = np.vstack((self.points, self.points[-1] + dR))
            self.steps = np.append(self.steps, step)

    def length(self):
        """Return length of the Branch."""
        return np.sum(np.linalg.norm(self.points[1:]-self.points[:-1], axis=1))

    def tip_angle(self):
        """Return the angle between the tip segment (last and penultimate point) and X axis."""
        point_penult = self.points[-2]
        point_last = self.points[-1]
        dx = point_last[0] - point_penult[0]
        dy = point_last[1] - point_penult[1]
        return np.arctan2(dy, dx)

    def points_steps(self):
        """Return a 3-n array of points and evolution steps when they were added."""
        return np.column_stack((self.points, self.steps))