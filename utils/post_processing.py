import pandas as pd
import numpy as np
from glob import glob
import os
from  deeplabcut.utils import auxiliaryfunctions,auxfun_multianimal
from scipy.interpolate import interp1d
from pathlib import Path
import deeplabcut
from deeplabcut.refine_training_dataset.outlier_frames import FitSARIMAXModel
from scipy import signal



def normalize(x):
    norm = np.linalg.norm(x, axis = 1)
    return x / norm[:, None]

def get_pos_mean(df, bodyparts):
    '''
    Given the trajectories of different bodyparts, get the average object postion.
    '''
    n_data = len(df)
    n_bodyparts = len(bodyparts)
    pos_mean = np.zeros((n_data, 3))
    for bp in bodyparts:
        pos_mean += df[bp].values
    return pos_mean/n_bodyparts


def get_ori_mean(df, bodyparts):
    '''
    Given the trajectories of different bodyparts, get the average object orientation.
    '''
    n_data = len(df)
    n_bodyparts = len(bodyparts)
    n_pairs = int(n_bodyparts/2)
    ori_mean = np.zeros((n_data, 3))

    for i in range(n_pairs):
        bp_h = bodyparts[i]
        bp_t = bodyparts[-(i+1)]
        if not bp_h == bp_t:
            ori_mean += normalize((df[bp_h].values - df[bp_t].values))

    return normalize(ori_mean)

# def interpolation_for_nan(df, window_size, kernel):
#     '''
#     Parameters:
#     x: the original trajectory data
#     window_size: the size of the moving window
#     kernel: The way to carry out the interpolation.
#             Options: 'x_linear': The velocity is assumed a constant and the trajectory will be linear.
#                      'v_linear': The velocity is assumed linear.
#                      'a_linear': The acceleration is assumed linear.
#
#     '''
#
#     ind = df.first_valid_index()
#     df[:ind] = df[ind]
#     x = df.values
#     x_p = np.zeros_like(x)
#     ind_not_non = np.argwhere(1 - np.isnan(x))
#     if kernel == 'x_linear':
#         for i, data in enumerate(x):
#             if not np.isnan(data):
#                 x_p[i] = x[i]
#             else:
#                 i_next = ind_not_non[(ind_not_non > i).argmax()]
#                 if not i_next == 0:
#                     v = (x[i_next] - x_p[i - 1])/(i_next - i + 1)
#                 else:
#                     v = x_p[i - 1] - x_p[i - 2]
#                 x_p[i] = x_p[i - 1] + v
#
#     #TODO: kernel == 'v_linear', 'a_linear' and 'a_p_linear'
#     return x_p


# def remove_nans(df, remove_method = 'interpolation', kernel = 'x_linear', window_size = 11):
#     '''
#     This function takes the file(obtained from trangulation) that containes the trajectories of the markers as input.
#     Return a dataframe without NaNs.
#
#     file_path: path to the file that containes the 3D trajectories of the makers.
#
#     nan_method: method to deal with the Nans in the dataframe. Options: 'drop', 'fill'
#                 'drop': This will just drop all the rows that contain Nans
#                 'fill': This will fill the Nans from previous non-Nan value if applicable. If there is no previous
#                         non-Nan value, it will fill the Nans from the following non-Nan value.
#                 'ignore': Do nothing.
#                 'interpolation': Use interpolation to fill the NaNs.
#     '''
#     if isinstance(df, str):
#         df = pd.read_hdf(df)
#     df_new = pd.DataFrame().reindex_like(df)
#
#
#     if remove_method == 'fill':
#         df = df.fillna(method = 'ffill').fillna(method = 'bfill') # get rid of NaNs
#     elif remove_method == 'drop':
#         df = df.dropna()
#     elif remove_method == 'ignore':
#         pass
#     elif remove_method =='interpolation':
#         for column in df.columns:
#             data = df[column]
#             df_new[column] = interpolation_for_nan(data, window_size, kernel)
#         return df_new
#     return df

# def remove_nans_and_save(h5file, remove_method = 'interpolation', kernel = 'x_linear', window_size = 5):
#     '''
#     This function takes the .h5 file that containes detections of deeplabcut. Fill the nans and then save it.
#
#     h5file: path to the file that containes detections of deeplabcut.
#
#     nan_method: method to deal with the Nans in the dataframe. Options: 'drop', 'fill'
#                 'drop': This will just drop all the rows that contain Nans
#                 'fill': This will fill the Nans from previous non-Nan value if applicable. If there is no previous
#                         non-Nan value, it will fill the Nans from the following non-Nan value.
#                 'ignore': Do nothing.
#                 'interpolation': Use interpolation to fill the NaNs.
#     '''
#     df = pd.read_hdf(h5file)
#
#     dirname = os.path.dirname(h5file)
#     camera = os.path.basename(os.path.dirname(h5file))
#     vid_id = os.path.basename(os.path.dirname(dirname))
#     if remove_method != 'interpolation':
#         outputname = dirname+ '/' + vid_id + '-' + camera + '_' + remove_method +'.h5'
#     else:
#         outputname = dirname+ '/' + vid_id + '-' + camera + '_' + kernel +'.h5'
#
#     df_new = pd.DataFrame().reindex_like(df)
#
#
#     if remove_method == 'fill':
#         df = df.fillna(method = 'ffill').fillna(method = 'bfill') # get rid of NaNs
#     elif remove_method == 'drop':
#         df = df.dropna()
#     elif remove_method == 'ignore':
#         pass
#     elif remove_method =='interpolation':
#         for column in df.columns:
#             data = df[column]
#             df_new[column] = interpolation_for_nan(data, window_size, kernel)
#         df_new.to_hdf(outputname, key ='df_without_nans')
#
#         return df_new, outputname
#     df.to_hdf(outputname, key ='df_without_nans')
#     return df_new, outputname

def get_obj_trajectories(file_path, remove_method = 'interpolation'):

    '''
    This function will take the file(obtained from trangulation) that containes the trajectories of the markers.
    Markers of the same object will be used to compute the position and orientation of each object at each frame.

    file_path: path to the file that containes the 3D trajectories of the makers.

    nan_method: method to deal with the Nans in the dataframe. Options: 'drop', 'fill'
                'drop': This will just drop all the rows that contain Nans
                'fill': This will fill the Nans from previous non-Nan value if applicable. If there is no previous
                        non-Nan value, it will fill the Nans from the following non-Nan value.
    '''

    df_with_nans = pd.read_hdf(file_path)

    df = remove_nans(df_with_nans, remove_method = remove_method)

    scorer = df.columns.get_level_values('scorer').unique()[0]
    individuals = df.columns.get_level_values('individuals').unique()
    df_new = pd.DataFrame()


    for individual in individuals:
        bodyparts = df[scorer][individual].columns.get_level_values('bodyparts').unique()
        n_bp = len(bodyparts)
        if individual != 'single':
            # this is an individual
            df_individual = df[scorer][individual]
            pos_mean = get_pos_mean(df_individual, bodyparts)
            ori_mean = get_ori_mean(df_individual, bodyparts)
            pose_mean = np.concatenate((pos_mean, ori_mean), axis = 1)
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
                pos_mean = get_pos_mean(df_individual, bodyparts_u)
                ori_mean = get_ori_mean(df_individual, bodyparts_u)
                pose_mean = np.concatenate((pos_mean, ori_mean), axis = 1)
                pdindex = pd.MultiIndex.from_product(
                            [[unique], ["x", "y", "z", "X", "Y","Z"]],
                            names=["individuals","pose"],
                        )
                frame = pd.DataFrame(pose_mean, columns=pdindex)
                df_new = pd.concat([frame, df_new], axis=1)
    df_new['time_stamp'] = df.index
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
                    data.loc[:, mask] = df.loc[:, mask].apply(
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
