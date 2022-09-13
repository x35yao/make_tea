"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d
matplotlib_axes_logger.setLevel("ERROR")

def triangulate(
    config3d,
    h5_left,
    h5_right,
    P1,
    P2,
    F,
    destfolder=None,
    save_as_csv=True,

):
    '''
    This function triangulate the DLC predictions from 2 cameras to obtain the 3d position of the makers on the objects.

    Parameters
    ----------
    config3d: str
        The path to the config file of the 3D DLC project.
    h5_left: str
        The path to the DLC prediction of the left camera.
    h5_right: str
        The path to the DLC prediction of the right camera.
    destfolder: str
        The path where the 3D prediction will be saved at.
    save_as_csv: bool
        Whether or not as the result as a csv file as well.
    P1: ndarray
        The projection matrix of the left camera
    P2: ndarray
        The projection matrix of the right camera
    F: ndarray
        The fundamental matrix.

    Returns
    -------
    df_3d: Dataframe
        A dataframe that contains the 3d predictions.

    '''
    cfg_3d = auxiliaryfunctions.read_config(config3d)

    cam_names = cfg_3d["camera_names"]
    pcutoff = cfg_3d["pcutoff"]
    scorer_3d = cfg_3d["scorername_3d"]
    snapshots = {}
    for cam in cam_names:
        snapshots[cam] = cfg_3d[str("config_file_" + cam)]
        # Check if the config file exists
        if not os.path.exists(snapshots[cam]):
            raise Exception(
                str(
                    "It seems the file specified in the variable config_file_"
                    + str(cam)
                )
                + " does not exist. Please edit the config file with correct file path and retry."
            )

    dataFrame_camera1_undistort = pd.read_hdf(h5_left)
    dataFrame_camera2_undistort = pd.read_hdf(h5_right)


    num_frames = np.min([dataFrame_camera1_undistort.shape[0], dataFrame_camera2_undistort.shape[0]])

    ### Deal with the case where h5_left and h5_right has different length. TODO: Might not be the best way to do

    dataFrame_camera1_undistort = dataFrame_camera1_undistort[:num_frames]
    dataFrame_camera2_undistort = dataFrame_camera2_undistort[:num_frames]

    scorer_cam1 = dataFrame_camera1_undistort.columns.get_level_values(0)[0]
    scorer_cam2 = dataFrame_camera2_undistort.columns.get_level_values(0)[0]

    bodyparts = dataFrame_camera1_undistort.columns.get_level_values(
        "bodyparts"
    ).unique()

    # print("Computing the triangulation...")

    ### Assign nan to [X,Y] of low likelihood predictions ###
    # Convert the data to a np array to easily mask out the low likelihood predictions
    if num_frames == 1: # This is for getting the template coords
        dataFrame_temp1 = dataFrame_camera1_undistort.copy()
        dataFrame_temp2 = dataFrame_camera2_undistort.copy()
        columns = []
        for name in dataFrame_temp1.columns.names:
            if not name == 'coords':
                values = dataFrame_temp1.columns.get_level_values(level = name).unique()
            else:
                values = pd.Index(['x', 'y', 'likelihood'], name = 'coords')
            columns.append(values)
        columns_new = pd.MultiIndex.from_product(columns)
        idx = pd.IndexSlice
        dataFrame_camera1_undistort, dataFrame_camera2_undistort = pd.DataFrame(columns= columns_new), pd.DataFrame(columns= columns_new)
        dataFrame_camera1_undistort.loc[:, idx[:, :, :, ['x']]] = dataFrame_temp1.loc[:, idx[:, :, :, ['x']]].to_numpy()
        dataFrame_camera1_undistort.loc[:, idx[:, :, :, ['y']]] = dataFrame_temp1.loc[:, idx[:, :, :, ['y']]].to_numpy()
        dataFrame_camera1_undistort.loc[:, idx[:, :, :, 'likelihood']] = 1

        dataFrame_camera2_undistort.loc[:, idx[:, :, :, ['x']]] = dataFrame_temp2.loc[:, idx[:, :, :, ['x']]].to_numpy()
        dataFrame_camera2_undistort.loc[:, idx[:, :, :, ['y']]] = dataFrame_temp2.loc[:, idx[:, :, :, ['y']]].to_numpy()
        dataFrame_camera2_undistort.loc[:, idx[:, :, :, 'likelihood']] = 1
    data_cam1_tmp = dataFrame_camera1_undistort.to_numpy().reshape(
        (num_frames, -1, 3)
    )

    data_cam2_tmp = dataFrame_camera2_undistort.to_numpy().reshape(
        (num_frames, -1, 3)
    )
    # Assign [X,Y] = nan to low likelihood predictions
    data_cam1_tmp[data_cam1_tmp[..., 2] < pcutoff, :2] = np.nan
    data_cam2_tmp[data_cam2_tmp[..., 2] < pcutoff, :2] = np.nan

    # Reshape data back to original shape
    data_cam1_tmp = data_cam1_tmp.reshape(num_frames, -1)
    data_cam2_tmp = data_cam2_tmp.reshape(num_frames, -1)
    # put data back to the dataframes
    dataFrame_camera1_undistort[:] = data_cam1_tmp
    dataFrame_camera2_undistort[:] = data_cam2_tmp

    config_2d = snapshots[cam_names[0]]
    cfg = auxiliaryfunctions.read_config(config_2d)
    if cfg.get("multianimalproject"):
        # Check individuals are the same in both views
        individuals_view1 = (
            dataFrame_camera1_undistort.columns.get_level_values("individuals")
            .unique()
            .to_list()
        )
        individuals_view2 = (
            dataFrame_camera2_undistort.columns.get_level_values("individuals")
            .unique()
            .to_list()
        )
        if individuals_view1 != individuals_view2:
            raise ValueError(
                "The individuals do not match between the two DataFrames"
            )

        # Cross-view match individuals
        _, voting = auxiliaryfunctions_3d.cross_view_match_dataframes(
            dataFrame_camera1_undistort, dataFrame_camera2_undistort, F
        )
    else:
        # Create a dummy variables for single-animal
        individuals_view1 = ["indie"]
        voting = {0: 0}

    # Cleaner variable (since inds view1 == inds view2)
    individuals = individuals_view1

    # Reshape: (num_framex, num_individuals, num_bodyparts , 2)
    all_points_cam1 = dataFrame_camera1_undistort.to_numpy().reshape(
        (num_frames, len(individuals), -1, 3)
    )[..., :2]
    all_points_cam2 = dataFrame_camera2_undistort.to_numpy().reshape(
        (num_frames, len(individuals), -1, 3)
    )[..., :2]

    # Triangulate data
    triangulate = []
    for k, _ in enumerate(individuals):
        # i is individual in view 1
        # voting[i] is the matched individual in view 2

        pts_indv_cam1 = all_points_cam1[:, k].reshape((-1, 2)).T
        pts_indv_cam2 = all_points_cam2[:, voting[k]].reshape((-1, 2)).T

        indv_points_3d = auxiliaryfunctions_3d.triangulatePoints(
            P1, P2, pts_indv_cam1, pts_indv_cam2
        )

        indv_points_3d = indv_points_3d[:3].T.reshape((num_frames, -1, 3))

        triangulate.append(indv_points_3d)

    triangulate = np.asanyarray(triangulate)

    # Create 3D DataFrame column and row indices
    axis_labels = ("x", "y", "z")
    if cfg.get("multianimalproject"):
        columns = pd.MultiIndex.from_product(
            [[scorer_3d], individuals, bodyparts, axis_labels],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

    else:
        columns = pd.MultiIndex.from_product(
            [[scorer_3d], bodyparts, axis_labels],
            names=["scorer", "bodyparts", "coords"],
        )

    inds = range(num_frames)
    # Swap num_animals with num_frames axes to ensure well-behaving reshape
    triangulate = triangulate.swapaxes(0, 1).reshape((num_frames, -1))

    # Fill up 3D dataframe
    df_3d = pd.DataFrame(triangulate, columns=columns, index=inds)
    if num_frames == 1:
        # This if for building object templates
        return df_3d
    vid_id = os.path.basename(h5_left).split('-')[0]
    if destfolder == None:
        destfolder = os.path.dirname(h5_left)
    output_filename = os.path.join(destfolder, vid_id + '_' + scorer_3d)
    if not os.path.isdir(destfolder):
        os.makedirs(destfolder)
    df_3d.to_hdf(
        str(output_filename + ".h5"),
        "df_with_missing",
        format="table",
        mode="w",
    )

    # Reorder 2D dataframe in view 2 to match order of view 1
    if cfg.get("multianimalproject"):
        df_2d_view2 = pd.read_hdf(h5_right)
        individuals_order = [individuals[i] for i in list(voting.values())]
        df_2d_view2 = auxfun_multianimal.reorder_individuals_in_df(df_2d_view2, individuals_order)
        df_2d_view2.to_hdf(h5_right, "tracks", format="table", mode="w",)


    if save_as_csv:
        df_3d.to_csv(str(output_filename + ".csv"))
    return df_3d

def _undistort_points(points, mat, coeffs, p, r):
    pts = points.reshape((-1, 3))
    pts_undist = cv2.undistortPoints(
        src=pts[:, :2].astype(np.float32),
        cameraMatrix=mat,
        distCoeffs=coeffs,
        P=p,
        R=r,
    )
    pts[:, :2] = pts_undist.squeeze()
    return pts.reshape((points.shape[0], -1))


def _undistort_views(df_view_pairs, stereo_params):
    df_views_undist = []
    for df_view_pair, camera_pair in zip(df_view_pairs, stereo_params):
        params = stereo_params[camera_pair]
        dfs = []
        for i, df_view in enumerate(df_view_pair, start=1):
            pts_undist = _undistort_points(
                df_view.to_numpy(),
                params[f"cameraMatrix{i}"],
                params[f"distCoeffs{i}"],
                params[f"P{i}"],
                params[f"R{i}"],
            )
            df = pd.DataFrame(pts_undist, df_view.index, df_view.columns)
            dfs.append(df)
        df_views_undist.append(dfs)
    return df_views_undist


def undistort_points(config, dataframe, camera_pair):
    cfg_3d = auxiliaryfunctions.read_config(config)
    path_camera_matrix = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)[2]
    """
    path_undistort = destfolder
    filename_cam1 = Path(dataframe[0]).stem
    filename_cam2 = Path(dataframe[1]).stem
    #currently no interm. saving of this due to high speed.
    # check if the undistorted files are already present
    if os.path.exists(os.path.join(path_undistort,filename_cam1 + '_undistort.h5')) and os.path.exists(os.path.join(path_undistort,filename_cam2 + '_undistort.h5')):
        print("The undistorted files are already present at %s" % os.path.join(path_undistort,filename_cam1))
        dataFrame_cam1_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam1 + '_undistort.h5'))
        dataFrame_cam2_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam2 + '_undistort.h5'))
    else:
    """
    # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
    dataframe_cam1 = pd.read_hdf(dataframe[0])
    dataframe_cam2 = pd.read_hdf(dataframe[1])
    path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
    stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)
    dataFrame_cam1_undistort, dataFrame_cam2_undistort = _undistort_views(
        [(dataframe_cam1, dataframe_cam2)], stereo_file,
    )[0]

    return (
        dataFrame_cam1_undistort,
        dataFrame_cam2_undistort,
        stereo_file[camera_pair],
        path_stereo_file,
    )