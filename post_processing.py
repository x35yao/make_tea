import pandas as pd
import numpy as np
from glob import glob
import os
from  deeplabcut.utils import auxiliaryfunctions,auxfun_multianimal
from scipy.interpolate import interp1d
from pathlib import Path
import deeplabcut
from deeplabcut.refine_training_dataset.outlier_frames import FitSARIMAXModel
from scipy import signal, ndimage
from sklearn.linear_model import LinearRegression,RANSACRegressor
from skimage.measure import LineModelND, ransac



def normalize(x):
    norm = np.linalg.norm(x, axis = 1)
    return x / norm[:, None]

def get_pos_mean(df, bodyparts, lr_model = 'ransac'):
    '''
    Given the trajectories of different bodyparts, get the average object postion.
    '''
    df_new = df.copy()
    if lr_model == 'ransac':
        n_data = len(df)
        pos = np.zeros((n_data, 3))
        for i in range(len(df)):
            data = df_new.iloc[i].values.reshape(-1, 3)
            X = data[:,:-1]
            y = data[:, -1]
            model_robust, inliers = ransac(data, LineModelND, min_samples=3,
                               residual_threshold=.6, max_trials=1000)
            print(i, inliers)
            outliers = inliers == False

            inds_in = np.where(inliers == 1)
            inds_out = np.where(outliers == 1)
            x = np.arange(data.shape[0])
            print(data)
            data[outliers, :] = np.nan

            spl_x = interp1d(x[inds_in], data[inds_in, 0],kind = 'linear', fill_value='extrapolate')
            spl_y = interp1d(x[inds_in], data[inds_in, 1],kind = 'linear', fill_value='extrapolate')
            spl_z = interp1d(x[inds_in], data[inds_in, 2],kind = 'linear', fill_value='extrapolate')

            x_interp = spl_x(inds_out).flatten()
            y_interp = spl_y(inds_out).flatten()
            z_interp = spl_z(inds_out).flatten()

            data[inds_out, 0] = x_interp
            data[inds_out, 1] = y_interp
            data[inds_out, 2] = z_interp

            print([x_interp, y_interp, z_interp])
            df_new.iloc[i] = data.flatten()

    pos_mean = df_new.mean(axis = 1, level = 'coords')
    pos_mean = np.nan_to_num(pos_mean)
    return pos_mean


def get_ori_mean(df, bodyparts, lr_model = 'ransac'):
    '''
    Given the trajectories of different bodyparts, get the average object orientation.
    '''
    n_data = len(df)
    n_bodyparts = len(bodyparts)
    n_pairs = int(n_bodyparts/2)
    ori = np.zeros((n_data, 3))
    for i in range(len(df)):
        data = df.iloc[i].values.reshape(-1, 3)
        X = data[:,:-1]
        y = data[:, -1]
        if lr_model == 'ransac':
            model_robust, inliers = ransac(data, LineModelND, min_samples=3,
                               residual_threshold=1.5, max_trials=1000)
            print(i, inliers)
            # if i < 10:
            #     print(inliers)
            # else:
            #     raise

        elif lr_model == 'linear':
            lr = LinearRegression()
            lr.fit(X, y)
            m = lr.coef_
        vec = model_robust.params[1]
        vec_normalized = vec / np.linalg.norm(vec)
        if i > 0:
            if np.dot(vec_normalized, ori[i-1,:]) < 0:
                # This is the situation when 2 consecutive frames have opposite direction
                vec_normalized = -1 * vec_normalized
        ori[i,:] = vec_normalized
    return ori

def get_pose_mean(df, bodyparts, lr_model = 'ransac', residual = 1):

    df_new = df.copy()
    if lr_model == 'ransac':
        n_data = len(df)
        ori_mean = np.zeros((n_data, 3))
        for i in range(len(df)):
            data = df_new.iloc[i].values.reshape(-1, 3)
            X = data[:,:-1]
            y = data[:, -1]
            model_robust, inliers = ransac(data, LineModelND, min_samples=3,
                               residual_threshold= residual, max_trials=1000)
            # print(i, inliers)s
            # print(data)
            outliers = inliers == False
            data[outliers, :] = np.nan
            inds_in = np.where(inliers == 1)
            inds_out = np.where(outliers == 1)
            x = np.arange(data.shape[0])

            spl_x = interp1d(x[inds_in], data[inds_in, 0],kind = 'linear', fill_value='extrapolate')
            spl_y = interp1d(x[inds_in], data[inds_in, 1],kind = 'linear', fill_value='extrapolate')
            spl_z = interp1d(x[inds_in], data[inds_in, 2],kind = 'linear', fill_value='extrapolate')

            x_interp = spl_x(inds_out).flatten()
            y_interp = spl_y(inds_out).flatten()
            z_interp = spl_z(inds_out).flatten()
            # print(np.c_[spl_x(inds_out), spl_y(inds_out), spl_z(inds_out)])
            # print(model_robust.params[1])
            data[inds_out, 0] = x_interp
            data[inds_out, 1] = y_interp
            data[inds_out, 2] = z_interp
            df_new.iloc[i] = data.flatten()

            vec = model_robust.params[1]
            vec_normalized = vec / np.linalg.norm(vec)
            if i > 0:
                if np.dot(vec_normalized, ori_mean[i-1,:]) < 0:
                    # This is the situation when 2 consecutive frames have opposite direction
                    vec_normalized = -1 * vec_normalized
            ori_mean[i,:] = vec_normalized
    pos_mean = df_new.mean(axis = 1, level = 'coords')
    pos_mean = np.nan_to_num(pos_mean)
    pose_mean = np.concatenate((pos_mean, ori_mean), axis = 1)
    return pose_mean

def filter_3D_data(file_path, filtertype = 'median', window_size = 5):

    df= pd.read_hdf(file_path)
    # df = remove_nans(df_with_nans, remove_method = remove_method)
    df_filtered = df.copy()

    if filtertype == 'median':
        f = signal.medfilt
    elif filtertype == 'gaussian':
        f = ndimage.gaussian_filter
    mask = df.columns.get_level_values("coords") != "likelihood"

    df_filtered.loc[:, mask] = df.loc[:, mask].apply(
                            f, args=(window_size,), axis=0)
    outputname = file_path.replace('.h5', f'_filtered_{filtertype}.h5')
    df_filtered.to_hdf(outputname, key = 'filtered_3d')
    return df_filtered


def get_obj_pose(file_path, residual = 1):

    '''
    This function will take the file(obtained from trangulation) that containes the trajectories of the markers.
    Markers of the same object will be used to compute the position and orientation of each object at each frame.

    file_path: path to the file that containes the 3D trajectories of the makers.

    nan_method: method to deal with the Nans in the dataframe. Options: 'drop', 'fill'
                'drop': This will just drop all the rows that contain Nans
                'fill': This will fill the Nans from previous non-Nan value if applicable. If there is no previous
                        non-Nan value, it will fill the Nans from the following non-Nan value.
    '''

    df= pd.read_hdf(file_path)
    # df = remove_nans(df_with_nans, remove_method = remove_method)
    df_filtered = df.copy()
    mask = df.columns.get_level_values("coords") != "likelihood"

    df_filtered.loc[:, mask] = df.loc[:, mask].apply(
                            signal.medfilt, args=(5,), axis=0)

    scorer = df.columns.get_level_values('scorer').unique()[0]
    individuals = df.columns.get_level_values('individuals').unique()
    df_new = pd.DataFrame()

    for individual in individuals:
        bodyparts = df[scorer][individual].columns.get_level_values('bodyparts').unique()
        n_bp = len(bodyparts)
        if individual != 'single':
            # this is an individual
            df_individual = df_filtered[scorer][individual]

            # pos_mean = get_pos_mean(df_individual, bodyparts)
            # ori_mean = get_ori_mean(df_individual, bodyparts)
            # pose_mean = np.concatenate((pos_mean, ori_mean), axis = 1)
            pose_mean = get_pose_mean(df_individual, bodyparts, lr_model = 'ransac', residual = residual)

            pdindex = pd.MultiIndex.from_product(
                        [[individual], ["x", "y", "z", "X", "Y","Z"]],
                        names=["individuals","pose"],
                    )
            frame = pd.DataFrame(pose_mean, columns=pdindex)
            df_new = pd.concat([frame, df_new], axis=1)
        elif individual == 'single':
            # these are the unique objects(pitcher, cup etc there is only one of which in the scene)
            uniques = set([bp.split('_')[0] for bp in bodyparts])
            for unique in uniques:
                bodyparts_u = [bp for bp in bodyparts if bp.split('_')[0] == unique]
                df_individual = df[scorer][individual]
                # ori_mean = get_ori_mean(df_individual, bodyparts_u)
                # pos_mean = get_pos_mean(df_individual, bodyparts_u)
                # pose_mean = np.concatenate((pos_mean, ori_mean), axis = 1)
                pose_mean = get_pose_mean(df_individual, bodyparts_u, lr_model = 'ransac')
                pdindex = pd.MultiIndex.from_product(
                            [[unique], ["x", "y", "z", "X", "Y","Z"]],
                            names=["individuals","pose"],
                        )
                frame = pd.DataFrame(pose_mean, columns=pdindex)
                df_new = pd.concat([frame, df_new], axis=1)
    df_new['time_stamp'] = df.index
    dirname = os.path.dirname(file_path)
    suffix = os.path.basename(file_path).split('_')[-1]
    if suffix == 'leastereo.h5':
        filename = dirname + '/obj_pose.h5'
    else:
        filename = dirname + f'/obj_pose_{suffix}'
    df_new.to_hdf(filename, key = 'obj_pose')
    return df_new

def columnwise_interp(data, filtertype, max_gap=0):
    """
    Perform cubic spline interpolation over the columns of *data*.
    All gaps of size lower than or equal to *max_gap* are filled,
    and data slightly smoothed.

    Parameters
    ----------
    data : array_like
        2D matrix of data.
    max_gap : int, optional
        Maximum gap size to fill. By default, all gaps are interpolated.

    Returns
    -------
    interpolated data with same shape as *data*
    """
    if np.ndim(data) < 2:
        data = np.expand_dims(data, axis=1)
    nrows, ncols = data.shape
    temp = data.copy()
    valid = ~np.isnan(temp)

    x = np.arange(nrows)
    for i in range(ncols):
        mask = valid[:, i]
        if (
            np.sum(mask) > 3
        ):  # Make sure there are enough points to fit the cubic spline
            spl = interp1d(x[mask], temp[mask, i],kind = filtertype, fill_value='extrapolate')
            y = spl(x)
            if max_gap > 0:
                inds = np.flatnonzero(np.r_[True, np.diff(mask), True])
                count = np.diff(inds)
                inds = inds[:-1]
                to_fill = np.ones_like(mask)
                for ind, n, is_nan in zip(inds, count, ~mask[inds]):
                    if is_nan and n > max_gap:
                        to_fill[ind : ind + n] = False
                y[~to_fill] = np.nan
            # Get rid of the interpolation beyond the spline knots
            y[y == 0] = np.nan
            temp[:, i] = y
    return temp

def interpolate_data(config,
    videos,
    videotype="mp4",
    shuffle=1,
    trainingsetindex=0,
    filtertypes="linear",
    windowlengths=0,
    p_bound=0.001,
    ARdegree=3,
    MAdegree=1,
    alpha=0.01,
    save_as_csv=False,
    destfolder=None,
    modelprefix="",
    track_method="",):

    cfg = auxiliaryfunctions.read_config(config)
    track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        trainFraction=cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    outputnames = []
    for video in videos:
        if destfolder is None:
            vid_folder = os.path.dirname(video)

        vname = Path(video).stem
        try:
            df, filepath, _, _ = auxiliaryfunctions.load_analyzed_data(
                vid_folder, vname, DLCscorer, track_method=track_method
            )
            data = df.copy()
            if not isinstance(filtertypes, list):
                outdataname = video.replace('.mp4', f'_{filtertype}.h5')
                filtertypes = [filtertypes]
            else:
                suffix = '_'.join(filtertypes)
                outdataname = video.replace('.mp4', f'_{suffix}.h5')
            for i, filtertype in enumerate(filtertypes):
                windowlength = windowlengths[i]
                print("Filtering with %s model %s" % (filtertype, video))
                if filtertype == 'arima':
                    temp = df.values.reshape((nrows, -1, 3))
                    placeholder = np.empty_like(temp)
                    for i in range(temp.shape[1]):
                        x, y, p = temp[:, i].T

                        meanx, _ = FitSARIMAXModel(
                            x, p, p_bound, alpha, ARdegree, MAdegree, False
                        )
                        meany, _ = FitSARIMAXModel(
                            y, p, p_bound, alpha, ARdegree, MAdegree, False
                        )
                        meanx[0] = x[0]
                        meany[0] = y[0]
                        placeholder[:, i] = np.c_[meanx, meany, p]
                    data = pd.DataFrame(
                        placeholder.reshape((nrows, -1)),
                        columns=df.columns,
                        index=df.index,
                    )
                elif filtertype == "median":
                    mask = data.columns.get_level_values("coords") != "likelihood"
                    data.loc[:, mask] = data.loc[:, mask].apply(
                        signal.medfilt, args=(windowlength,), axis=0
                    )
                else:
                    nrows = df.shape[0]
                    mask_data = data.columns.get_level_values("coords").isin(("x", "y"))
                    xy = data.loc[:, mask_data].values
                    prob = data.loc[:, ~mask_data].values
                    missing = np.isnan(xy)
                    xy_filled = columnwise_interp(xy, filtertype, windowlength)
                    filled = ~np.isnan(xy_filled)
                    xy[filled] = xy_filled[filled]
                    inds = np.argwhere(missing & filled)
                    if inds.size:
                        # Retrieve original individual label indices
                        inds[:, 1] //= 2
                        inds = np.unique(inds, axis=0)
                        prob[inds[:, 0], inds[:, 1]] = 0.01
                        data.loc[:, ~mask_data] = prob
                    data.loc[:, mask_data] = xy
            data.to_hdf(outdataname, "df_with_missing", format="table", mode="w")
            print(f'The h5 file is saved at: {outdataname}')
            if save_as_csv:
                print("Saving filtered csv poses!")
                data.to_csv(outdataname.split(".h5")[0] + ".csv")
            outputnames.append(outdataname)
        except FileNotFoundError as e:
            print(e)
            continue
    return outputnames


def combine_h5files(objs, video, suffix, to_csv = True):
    df_new = pd.DataFrame
    video_dir = os.path.dirname(video)
    for obj in objs:
        h5file = glob(video_dir + '/' + f'*obj*_{suffix}.h5')
        df = pd.read_hdf(h5file)
        pd.concat([df_new, df], axis = 1)
    df_new.to_hdf('combined.h5', key = 'combined_data')
    if to_csv:
        df_new.to_csv('combined.csv')


# def create_video_without_nans(config, video, h5file, remove_method, kernel = 'x_linear', window_size = 3):
#     df_new, newh5file = remove_nans_and_save(h5file, remove_method = remove_method, kernel = kernel, window_size = window_size)
#     if remove_method != 'interpolation':
#         surfix = remove_method
#     else:
#         surfix = kernel
#     create_video_with_h5file(config, video, newh5file, surfix = surfix)
