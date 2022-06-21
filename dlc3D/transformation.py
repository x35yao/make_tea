import numpy as np

def build_homogeneous_matrix(rotmatrix, translation):
    H = np.zeros((3, 4))
    H[:3, :3] = rotmatrix
    H[:, 3] = translation
    return H

def camera_matrix_to_fundamental_matrix(camera_matrix_1, camera_matrix_2, R, t):
    A  = camera_matrix_1 @ R.T @ t
    C = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
    F = np.linalg.inv(camera_matrix_2).T @ R @ camera_matrix_1.T @ C

    return F