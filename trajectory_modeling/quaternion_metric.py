import numpy as np
from numpy.linalg import norm as l2norm
from scipy import ndimage


def force_rotation_axis_direction(quat, axis=[1, 0, 0]):
    vect = quat[:3]
    dot_product = vect @ axis
    if dot_product > 0:
        return quat
    else:
        return -quat


def force_smooth_quats( quats):
    for i, quat_current in enumerate(quats):
        if i == 0:
            pass
        else:
            quat_previous = quats[i - 1]
            result = quat_current[:3] @ quat_previous[:3]
            if result > 0:
                quats[i] = quat_current
            else:
                quats[i] = -quat_current
    return quats


def process_quaternions(quats, sigma = None):
    init_quat = force_rotation_axis_direction(quats[0])
    quats[0] = init_quat
    quats_new = force_smooth_quats(quats)
    if sigma != None:
        quats_filtered = ndimage.gaussian_filter(quats_new, sigma=sigma)
        return quats_filtered
    else:
        return quats_new


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