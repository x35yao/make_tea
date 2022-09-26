import numpy as np
from numpy.linalg import norm as l2norm


def norm_diff_quat(q1, q2):
    """
    return a distance metric measure between q1 and q2 quaternion based on the Norm of Difference Quaternions metric.
    Parameters:
    -----------
    q1: np.array
        A array of 4 float representation of a quaternion.
    q2: np.array
        A array of 4 float representation of a quaternion.
    Returns:
    --------
    distance: float
    """
    q1 = q1 / l2norm(q1)
    q2 = q2 / l2norm(q2)
    return np.min([l2norm(q1 + q2), l2norm(q1 - q2)])

def inner_prod_quat(q1, q2):
    """
    return a distance metric measure between q1 and q2 quaternion based on the Inner Product of unit Quaternions metric.
    Parameters:
    -----------
    q1: np.array
        A array of 4 float representation of a quaternion.
    q2: np.array
        A array of 4 float representation of a quaternion.
    Returns:
    --------
    distance: float
        in radian
    """
    q1 = q1/l2norm(q1)
    q2 = q2/l2norm(q2)
    return np.arccos(np.abs(q1.dot(q2)))