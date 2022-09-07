import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import os
import pickle


def homogenous_position(vect):
    '''
    This function will make sure the coordinates in the right shape(4 by N) that could be multiplied by a
    homogeneous transformation(4 by 4).

    Parameters
    ----------
    vect: list or np.array
        A N by 3 array. x, y, z coordinates of the N points

    Return
    ------
    A 4 by N array.
    '''
    temp = np.array(vect)
    if temp.ndim == 1:
        ones = np.ones(1)
        return np.r_[temp, ones].reshape(-1, 1)
    elif temp.ndim == 2:
        num_rows, num_cols = temp.shape
        if num_cols != 3:
            raise Exception(f"vect is not N by 3, it is {num_rows}x{num_cols}")

        ones = np.ones(num_rows).reshape(-1, 1)
        return np.c_[temp, ones].T

def lintrans(a, H):
    '''
    This function applies the linear transformation expressed with the homogeneous transformation H.

    Parameters:
    -----------
    a: np.array
        The trajectory data with shape (N by D) where N is the number of datapoints and D is the dimension.
    H: np.array
        A 4 by 4 homogeneous transformation matrix.

    Returns:
    --------
    a_transformed: np.array
        A N by D array that contains the transformed trajectory
    '''
    D = a.shape[1]
    if D == 3: # position only
        a_homo = homogenous_position(a)
        a_transformed = (H @ a_homo).T[:,:3]
    elif D == 7: # position + orientation(quaternion)
        a_homo = homogenous_position(a[:,:3])
        pos = (H @ a_homo).T[:,:3]
        rot1 = R.from_matrix(H[:3, :3])
        rot2 = R.from_quat(a[:, 3:])
        rot = rot1 * rot2
        ori = rot.as_quat()
        a_transformed = np.concatenate((pos, ori), axis = 1)
    return a_transformed

def rigid_transform_3D(A, B):
    '''
    This function finds the rotation and translation that matches A to B in the same reference frame.

    Parameters
    ----------
    A: array
        N * 3 array, where N is the number of datapoints
    B: array
        N * 3 array, where N is the number of datapoints

    Returns
    -------
    R: array
        3 * 3 array, the rotation matrix to match A with B
    t: array
        3 array, the translation that match A with B

    '''
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_cols != 3:
        raise Exception(f"matrix A is not Nx3, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_cols != 3:
        raise Exception(f"matrix B is not Nx3, it is {num_rows}x{num_cols}")
    A = A.T
    B = B.T
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

#     special reflection case
    if np.linalg.det(R) < 0:
#         print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def homogenous_transform(R,vect):

    '''
    :param R: 3x3 matrix
    :param vect: list x,y,z
    :return:Homogenous transformation 4x4 matrix using R and vect
    '''
    if not isinstance(vect, list):
        vect = list(vect)
    H = np.zeros((4,4))
    H[0:3,0:3] = R
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

def get_HT_template_in_obj(template, markers_coords_in_camera, camera_in_template):
    '''
    This function will align the object with template and output the homogeneous transformation
    and the markers' average distance

    Parameters
    ----------
    template: dict
        The obj template that has the markers position in camera reference frame
    markers_coords_in_camera: dict or Dataframe
        The dict or Dataframe that contains the marker's coordinates in camera reference frame for a frame of the video
    camera_in_template: np.array
        A 4 by 4 array that represents camera in the template's reference frame

    Returns
    -------
    template_in_obj: np.array
        A 4 by 4 homogeneous transformation that represents template in the object's reference frame
    dist: float
        The lowest average markers' distance after aligning object with the template

    '''
    idx = pd.IndexSlice
    points_template = []
    points_obj = []
    if not isinstance(markers_coords_in_camera, dict):
        bps = markers_coords_in_camera.index.get_level_values('bodyparts').unique()
        for bodypart in bps:
            df_bodypart = markers_coords_in_camera.loc[bodypart]
            if not df_bodypart.isnull().values.any():
                points_template.append(template[bodypart])
                points_obj.append(df_bodypart.to_numpy())
    else:
        keys = markers_coords_in_camera.keys()
        for bodypart in keys:
            df_bodypart = markers_coords_in_camera[bodypart]
            if not df_bodypart.isnull().values.any():
                points_template.append(template[bodypart])
                points_obj.append(df_bodypart.to_numpy())

    points_template = np.array(points_template)
    points_obj = np.array(points_obj)

    points_template_in_template = lintrans(points_template, camera_in_template)
    points_obj_in_template = lintrans(points_obj, camera_in_template)
    rotmatrix, translation = rigid_transform_3D(points_obj_in_template, points_template_in_template)
    H = homogenous_transform(rotmatrix, list(translation.flatten()))
    points_obj_transformed = lintrans(points_obj_in_template, H)
    dists = []
    for i, point in enumerate(points_obj_transformed):
        point = point
        dist = np.linalg.norm(point- points_template_in_template[i])
        dists.append(dist)
    dist_average = np.mean(np.array(dists), axis=0)
    return H, dist_average

def match_markers_to_template(obj_template, df_individual, camera_in_template, window_size, n_marker=3):
    '''
    Search the first window_size rows of df_individual so that it matches the obj_template the best.

    Parameters
    ----------
    obj_template: dict
        The obj template that has the markers position in camera reference frame
    df_individual: Dataframe
        The Dataframe that contains the DLC + LEAstereo output for an individual
    camera_in_template: np.array
        A 4 by 4 array that represents camera in the template's reference frame
    window_size: int
        The first window_size rows that will be searched to find the best row
    n_marker: int
        The number of markers that will be used to align object with template.

    Returns
    -------
    template_in_obj: np.array
        A 4 by 4 homogeneous transformation that represents template in the object's reference frame
    dist: float
        The lowest average markers' distance after aligning object with the template

    '''
    dist = np.inf
    for i in range(window_size):
        marker_coords = df_individual.iloc[i]
        bps = marker_coords.index.get_level_values('bodyparts').unique()
        combs = combinations(bps, n_marker)
        for comb in combs:
            partial_marker_coords = marker_coords.loc[list(comb)]
            if partial_marker_coords.isnull().any():
                continue
            H, dist_average = get_HT_template_in_obj(obj_template, partial_marker_coords, camera_in_template)
            if dist_average < dist:
                dist = dist_average
                template_in_obj = H

    return template_in_obj, dist

def get_HT_template_in_camera(obj_template, ndi_in_camera):
    '''
    This function will assign a frames to the template where the x,y,z axes are aligned with NDI axes, and the origin will
    be at the mean of the markers of the template.

    Parameters:
    -----------
    obj_template: dict
        A dictionary that contains the makers' positions in camera's reference frame.
    ndi_in_camera: np.array
        A 4 by 4 array that contains the homogeneous transformation that expresses NDI in camera reference frame.

    Returns:
    --------
    template_in_camera: np.array
        A 4 by 4 array that contains the homogeneous transformation that expresses template in camera reference frame.
    '''
    rot_matrix = ndi_in_camera[:3, :3]
    markers = []
    for bp in obj_template:
        markers.append(obj_template[bp])
    markers = np.array(markers)
    center = np.mean(markers, axis=0)
    template_in_camera = homogenous_transform(rot_matrix, list(center))
    return template_in_camera

def get_HT_obj_in_ndi(obj_trajs_action, individuals, template_dir, thres = 20):
    '''
    This function will get rotation matrix A and translation b for each object in each demonstration.

    Parameters:
    -----------
    obj_trajs_action: dict
        A directory that contains the object markers trajectories in camera reference frame for different demonstrations.
    individuals: list
        A list that contains the objects relative to the task.
    template_dir: str
        The directory that contains the objects' template information.
    thres: int
        The threshold that determines whether or not the object is matched with the template

    Returns
    -------
    HTs: dict
        A dictionary that contains the homogeneous transformation for object in each demonstration
    bad_demos: list
        The list of demos that the average distance between the matched points are higher than the threshold.
    '''
    with open(os.path.join(template_dir, 'obj_templates.pickle'), "rb") as f:
        obj_templates = pickle.load(f)
    with open(os.path.join(template_dir, 'camera_to_ndi.pickle'), "rb") as f:
        camera_in_ndi = pickle.load(f)
    with open(os.path.join(template_dir, 'ndi_to_camera.pickle'), "rb") as f:
        ndi_in_camera = pickle.load(f)
    window_size = 20
    bad_demos = [] # the demos that the objects' markers fail to match with the template

    demos = obj_trajs_action.keys()
    HTs = {}
    for individual in individuals:
        HTs[individual] = {}
        if 'teabag' in individual:
            obj = 'teabag'
        elif individual == 'global':
            for demo in demos:
                # No transformation needed for global reference frame
                A = np.eye(3)
                b = np.zeros(3)
                H = homogenous_transform(A, b)
                HTs[individual][demo] = H
        else:
            obj = individual
        obj_template = obj_templates[obj]
        template_in_camera = get_HT_template_in_camera(obj_template, ndi_in_camera)
        camera_in_template = inverse_homogenous_transform(template_in_camera)
        for demo in demos:
            df_individual = obj_trajs_action[demo][individual]
            template_in_obj, dist = match_markers_to_template(obj_template, df_individual, camera_in_template,
                                                              window_size)
            if dist > thres:
                bad_demos.append(demo)
                continue
            obj_in_template = inverse_homogenous_transform(template_in_obj)
            obj_in_ndi = camera_in_ndi @ template_in_camera @ obj_in_template
            HTs[individual][demo] = obj_in_ndi
    return HTs, bad_demos