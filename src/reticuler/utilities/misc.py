"""Miscellaneous.

Labels for boundary conditions

Functions:
    find_reconnection_point
    cyl2cart
    cart2cyl
    extend_radially

Classes:
    NumpyEncoder
    
"""

import numpy as np
import json
import os

# Labels for boundary conditions
DIRICHLET_0 = 1 # u=0
DIRICHLET_1 = 2 # u=1
DIRICHLET_GLOB_FLUX = 3 # global flux constant (rescaled Dirichlet)
NEUMANN_0 = 4 # zero flux boundary condition
NEUMANN_1 = 5 # constant flux = -1 (influx)
RIGHT_WALL_PBC = 999
LEFT_WALL_PBC = 998

def find_reconnection_point(pt, starting_points, ending_points, too_close=0.1):
    """Cartesian distance from a point to line segment
    https://stackoverflow.com/a/58781995

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Parameters
    ----------
    pt : array
        A 2-n array
    starting_points : array
        A 2-n array with starting points of segments.
    ending_points : array
        A 2-n array with ending points of segments.
    too_close : float, default 0.1
        A threshold to determine if we insert new point on the closest segment.

    Returns
    -------
    distances : float
        The distance to the closest segment.
    ind_min : int
        Index of the closest segment.
    is_pt_new : bool
        Flag if new point on the closest segment should be inserted.
    breakthrough_pt : array
        Coordinates of the reconnection point (one of the segment ends or inserted).
    
    """
    # normalized tangent vectors
    d_ba = ending_points - starting_points
    d = d_ba / (np.linalg.norm(d_ba, axis=1).reshape(-1, 1))

    # signed distance of projection of the point on a segment to its ends
    # for a segment from [0,0] to [2,0] and a point [-1,1]:
    # s = 1 , t = -3
    s = np.sum( (starting_points - pt) * d, axis=1)
    t = np.sum( (pt - ending_points) * d, axis=1)

    # distance to the closest segment end if pt outside, 0 otherwise
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance: c = || d_pa x d || (cross product)
    d_pa = pt - starting_points
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    
    # closest points on the segments
    # we don't put new node if it's too close to the old ones
    # e = (s>t).reshape(3,1) * (a - ((s<-too_close)*s).reshape(3,1)*d) + \
    #     (s<t).reshape(3,1) * (b + ((t<-too_close)*t).reshape(3,1)*d)

    distances = np.sqrt( h**2 + c**2 )
    ind_min = np.argmin(distances)
    
    # we don't put new node if it's too close to the old ones
    s1 = s[ind_min]
    t1 = t[ind_min]
    if s1>t1:
        is_pt_new = s1<-too_close
        breakthrough_pt = starting_points[ind_min] - is_pt_new*s1*d[ind_min]
        ind_min_end = 0
    else:
        is_pt_new = t1<-too_close
        breakthrough_pt = ending_points[ind_min] + is_pt_new*t1*d[ind_min]
        ind_min_end = 1
    
    return distances[ind_min], ind_min, is_pt_new, breakthrough_pt, ind_min_end

def cyl2cart(r, theta, R0):
    # theta measured from the negative Y axis
    return np.array([R0+r*np.sin(theta), R0-r*np.cos(theta)]).T

def cart2cyl(x, y, R0):
    # theta measured from the negative Y axis
    return np.array([np.sqrt( (R0-x)**2 + (R0-y)**2 ), np.arctan2(y-R0,x-R0)+np.pi/2]).T

def extend_radially(pts, R0, beta):
    pts_cyl = cart2cyl(*pts.T, R0)
    pts_cyl[:,0] = pts_cyl[:,0] * beta
    pts1 = cyl2cart(*pts_cyl.T, R0*beta)
    return pts1

def rotation_matrix(angle):
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
    >>> rot = rotation_matrix(angle)
    >>> rotated_vector = np.dot(rot, vector)

    """
    return np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64): 
            return int(obj)
        return json.JSONEncoder.default(self, obj)