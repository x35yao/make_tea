"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from tqdm import tqdm

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d
from deeplabcut.utils import auxfun_multianimal

matplotlib_axes_logger.setLevel("ERROR")


def triangulate(
    config,
    h5files,
    destfolder=None,
    save_as_csv=False,
):
    """
    This function triangulates the detected DLC-keypoints from the two camera views
    using the camera matrices (derived from calibration) to calculate 3D predictions.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    video_path : string/list of list
        Full path of the directory where videos are saved. If the user wants to analyze
        only a pair of videos, the user needs to pass them as a list of list of videos,
        i.e. [['video1-camera-1.avi','video1-camera-2.avi']]

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n
        Only videos with this extension are analyzed. The default is ``.avi``

    filterpredictions: Bool, optional
        Filter the predictions with filter specified by "filtertype". If specified it
        should be either ``True`` or ``False``.

    filtertype: string
        Select which filter, 'arima' or 'median' filter (currently supported).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi).
        If you do not have a GPU put None.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video)

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``

    Example
    -------
    Linux/MacOS
    To analyze all the videos in the directory:
    >>> deeplabcut.triangulate(config,'/data/project1/videos/')

    To analyze only a few pairs of videos:
    >>> deeplabcut.triangulate(config,[['/data/project1/videos/video1-camera-1.avi','/data/project1/videos/video1-camera-2.avi'],['/data/project1/videos/video2-camera-1.avi','/data/project1/videos/video2-camera-2.avi']])


    Windows
    To analyze all the videos in the directory:
    >>> deeplabcut.triangulate(config,'C:\\yourusername\\rig-95\\Videos')

    To analyze only a few pair of videos:
    >>> deeplabcut.triangulate(config,[['C:\\yourusername\\rig-95\\Videos\\video1-camera-1.avi','C:\\yourusername\\rig-95\\Videos\\video1-camera-2.avi'],['C:\\yourusername\\rig-95\\Videos\\video2-camera-1.avi','C:\\yourusername\\rig-95\\Videos\\video2-camera-2.avi']])
    """
    from deeplabcut.pose_estimation_tensorflow import predict_videos
    from .post_processing import interpolate_data

    cfg_3d = auxiliaryfunctions.read_config(config)
    cam_names = cfg_3d["camera_names"]
    pcutoff = cfg_3d["pcutoff"]
    scorer_3d = cfg_3d["scorername_3d"]
    scorer_name = {}

    print("Undistorting...")
    (
        dataFrame_camera1_undistort,
        dataFrame_camera2_undistort,
        stereomatrix,
        path_stereo_file,
    ) = undistort_points(
        config, h5files, str(cam_names[0] + "-" + cam_names[1])
    )
    if len(dataFrame_camera1_undistort) != len(dataFrame_camera2_undistort):
        import warnings

        warnings.warn(
            "The number of frames do not match in the two videos. Please make sure that your videos have same number of frames and then retry! Excluding the extra frames from the longer video."
        )
        if len(dataFrame_camera1_undistort) > len(dataFrame_camera2_undistort):
            dataFrame_camera1_undistort = dataFrame_camera1_undistort[
                : len(dataFrame_camera2_undistort)
            ]
        if len(dataFrame_camera2_undistort) > len(dataFrame_camera1_undistort):
            dataFrame_camera2_undistort = dataFrame_camera2_undistort[
                : len(dataFrame_camera1_undistort)
            ]
    #                raise Exception("The number of frames do not match in the two videos. Please make sure that your videos have same number of frames and then retry!")

    scorer_cam1 = dataFrame_camera1_undistort.columns.get_level_values(0)[0]
    scorer_cam2 = dataFrame_camera2_undistort.columns.get_level_values(0)[0]
    df_3d, scorer_3d, bodyparts = auxiliaryfunctions_3d.create_empty_df(
        dataFrame_camera1_undistort, scorer_3d, flag="3d"
    )
    P1 = stereomatrix["P1"]
    P2 = stereomatrix["P2"]

    print("Computing the triangulation...")
    individuals = dataFrame_camera1_undistort.columns.get_level_values("individuals").unique()
    for individual in individuals:
        X_final = []
        triangulate = []
        bodyparts = dataFrame_camera1_undistort[scorer_cam1][individual].columns.get_level_values("bodyparts").unique()
        for bpindex, bp in enumerate(bodyparts):
            # Extract the indices of frames where the likelihood of a bodypart for both cameras are less than pvalue
            likelihoods = np.array(
                [
                    dataFrame_camera1_undistort[scorer_cam1][individual][bp][
                        "likelihood"
                    ].values[:],
                    dataFrame_camera2_undistort[scorer_cam2][individual][bp][
                        "likelihood"
                    ].values[:],
                ]
            )
            likelihoods = likelihoods.T

            # Extract frames where likelihood for both the views is less than the pcutoff
            low_likelihood_frames = np.any(likelihoods < pcutoff, axis=1)
            # low_likelihood_frames = np.all(likelihoods < pcutoff, axis=1)

            low_likelihood_frames = np.where(low_likelihood_frames == True)[0]
            points_cam1_undistort = np.array(
                [
                    dataFrame_camera1_undistort[scorer_cam1][individual][bp]["x"].values[:],
                    dataFrame_camera1_undistort[scorer_cam1][individual][bp]["y"].values[:],
                ]
            )
            points_cam1_undistort = points_cam1_undistort.T

            # For cam1 camera: Assign nans to x and y values of a bodypart where the likelihood for is less than pvalue
            points_cam1_undistort[low_likelihood_frames] = np.nan, np.nan
            points_cam1_undistort = np.expand_dims(points_cam1_undistort, axis=1)

            points_cam2_undistort = np.array(
                [
                    dataFrame_camera2_undistort[scorer_cam2][individual][bp]["x"].values[:],
                    dataFrame_camera2_undistort[scorer_cam2][individual][bp]["y"].values[:],
                ]
            )
            points_cam2_undistort = points_cam2_undistort.T

            # For cam2 camera: Assign nans to x and y values of a bodypart where the likelihood is less than pvalue
            points_cam2_undistort[low_likelihood_frames] = np.nan, np.nan
            points_cam2_undistort = np.expand_dims(points_cam2_undistort, axis=1)

            X_l = auxiliaryfunctions_3d.triangulatePoints(
                P1, P2, points_cam1_undistort, points_cam2_undistort
            )

            # ToDo: speed up func. below by saving in numpy.array
            X_final.append(X_l)
        triangulate.append(X_final)
        triangulate = np.asanyarray(triangulate)
        metadata = {}
        metadata["stereo_matrix"] = stereomatrix
        metadata["stereo_matrix_file"] = path_stereo_file
        # metadata["scorer_name"] = {
        #     cam_names[0]: scorer_name[cam_names[0]],
        #     cam_names[1]: scorer_name[cam_names[1]],
        # }

        # Create an empty dataframe to store x,y,z of 3d data
        for bpindex, bp in enumerate(bodyparts):
            df_3d.iloc[:][scorer_3d, individual, bp, "x"] = triangulate[0, bpindex, 0, :]
            df_3d.iloc[:][scorer_3d, individual, bp, "y"] = triangulate[0, bpindex, 1, :]
            df_3d.iloc[:][scorer_3d, individual, bp, "z"] = triangulate[0, bpindex, 2, :]
        if destfolder == None:
            destfolder = os.path.dirname(os.path.dirname(h5files[0]))
        filename =  os.path.basename(h5files[0])
        kernels = filename.split('.')[0].split('_')[1:]
        suffix = '_'.join(kernels)
        output_filename = destfolder + f'/make_tea3D_{suffix}'
        df_3d.to_hdf(
            str(output_filename + ".h5"),
            "df_with_missing",
            format="table",
            mode="w",
        )
        auxiliaryfunctions_3d.SaveMetadata3d(
            str(output_filename + "_meta.pickle"), metadata
        )

        if save_as_csv:
            df_3d.to_csv(str(output_filename + ".csv"))

        print("Results are saved under: ", destfolder)



"""
ToDo: speed up func. below and check only for one cam individually
PredicteData = np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]
pdindex = pd.MultiIndex.from_product([[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],names=['scorer', 'bodyparts', 'coords'])
auxiliaryfunctions.SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, framelist,save_as_csv)
"""


def undistort_points(config, dataframe, camera_pair):
    cfg_3d = auxiliaryfunctions.read_config(config)
    (
        img_path,
        path_corners,
        path_camera_matrix,
        path_undistort,
        _,
    ) = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)
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
    if True:
        # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
        dataframe_cam1 = pd.read_hdf(dataframe[0])
        dataframe_cam2 = pd.read_hdf(dataframe[1])
        scorer_cam1 = dataframe_cam1.columns.get_level_values(0)[0]
        scorer_cam2 = dataframe_cam2.columns.get_level_values(0)[0]
        path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
        stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)
        mtx_l = stereo_file[camera_pair]["cameraMatrix1"]
        dist_l = stereo_file[camera_pair]["distCoeffs1"]

        mtx_r = stereo_file[camera_pair]["cameraMatrix2"]
        dist_r = stereo_file[camera_pair]["distCoeffs2"]

        R1 = stereo_file[camera_pair]["R1"]
        P1 = stereo_file[camera_pair]["P1"]

        R2 = stereo_file[camera_pair]["R2"]
        P2 = stereo_file[camera_pair]["P2"]

        # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
        (
            dataFrame_cam1_undistort,
            scorer_cam1,
            bodyparts,
        ) = auxiliaryfunctions_3d.create_empty_df(
            dataframe_cam1, scorer_cam1, flag="2d"
        )
        (
            dataFrame_cam2_undistort,
            scorer_cam2,
            bodyparts,
        ) = auxiliaryfunctions_3d.create_empty_df(
            dataframe_cam2, scorer_cam2, flag="2d"
        )
        individuals = dataFrame_cam1_undistort.columns.get_level_values("individuals").unique()
        for individual in individuals:
            bodyparts = dataFrame_cam1_undistort[scorer_cam1][individual].columns.get_level_values("bodyparts").unique()
            for bpindex, bp in tqdm(enumerate(bodyparts)):
                points_cam1 = dataframe_cam1[scorer_cam1][individual][bp].values[:,:-1]
                points_cam1_remapped = cv2.undistortPoints(
                    src=points_cam1.astype(np.float32),
                    cameraMatrix=mtx_l,
                    distCoeffs=dist_l,
                    P=P1,
                    R=R1,
                )

                dataFrame_cam1_undistort.loc(axis=1)[
                    scorer_cam1, individual, bp, ["x", "y"]
                ] = points_cam1_remapped.squeeze()
                dataFrame_cam1_undistort.loc(axis=1)[
                    scorer_cam1, individual, bp, "likelihood"
                ] = dataframe_cam1.xs(
                    [individual, bp, "likelihood"], level=["individuals","bodyparts", "coords"], axis=1
                ).values

                # Undistorting the points from cam2 camera
                points_cam2 = dataframe_cam2[scorer_cam2][individual][bp].values[:,:-1]
                points_cam2_remapped = cv2.undistortPoints(
                    src=points_cam2.astype(np.float32),
                    cameraMatrix=mtx_r,
                    distCoeffs=dist_r,
                    P=P2,
                    R=R2,
                )
                dataFrame_cam2_undistort.loc(axis=1)[
                    scorer_cam2, individual, bp, ["x", "y"]
                ] = points_cam2_remapped.squeeze()

                dataFrame_cam2_undistort.loc(axis=1)[
                    scorer_cam2, individual, bp, "likelihood"
                ] = dataframe_cam2.xs(
                    [individual, bp, "likelihood"], level=["individuals", "bodyparts", "coords"], axis=1
                ).values


        # Save the undistorted files
        dataFrame_cam1_undistort.sort_index(inplace=True)
        dataFrame_cam2_undistort.sort_index(inplace=True)

    return (
        dataFrame_cam1_undistort,
        dataFrame_cam2_undistort,
        stereo_file[camera_pair],
        path_stereo_file,
    )
