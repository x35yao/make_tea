import numpy as np

def L2norm(q):
    """
    Calculate the l2 norm of an vector
    ----------
    q: list of float
    """
    return np.sum(np.power(q, 2))


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
    return np.min([L2norm(q1 + q2), L2norm(q1 - q2)])
    
    
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
    return np.arccos(np.abs(q1.dot(q2)))