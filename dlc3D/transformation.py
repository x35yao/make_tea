import numpy as np

def homogenous_transform(R,vect):

    '''
    :param R: 3x3 matrix
    :param vect: list x,y,z
    :return:Homogenous transformation 4x4 matrix using R and vect
    '''

    H = np.zeros((4,4))
    H[0:3,0:3] = R
    if not isinstance(vect, list):
        frame_displacement = list(vect) + [1]
    else:
        frame_displacement = vect + [1]
    D = np.array(frame_displacement)
    D.shape = (1,4)
    H[:,3] = D
    return H

def inverse_homogenous_transform(H):

    '''
    :param H: Homogenous Transform Matrix
    :return: Inverse Homegenous Transform Matrix
    '''


    R = H[0:3,0:3]
    origin = H[:-1,3]
    origin.shape = (3,1)

    R = R.T
    origin = -R.dot(origin)
    return homogenous_transform(R,list(origin.flatten()))

def homogenous_position(vect):
    if isinstance(vect, list):
        return np.array(vect + [1])
    else:
        return np.array(list(vect) + [1])

def camera_matrix_to_fundamental_matrix(camera_matrix_1, camera_matrix_2, R, t):
    A  = camera_matrix_1 @ R.T @ t
    C = np.array([[0, -A[2], A[1]], [A[2], 0, -A[0]], [-A[1], A[0], 0]])
    F = np.linalg.inv(camera_matrix_2).T @ R @ camera_matrix_1.T @ C

    return F