import pandas as pd
import numpy as np
from glob import glob

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

def interpolation_for_nan(df, window_size, kernel):
    '''
    Parameters:
    x: the original trajectory data
    window_size: the size of the moving window
    kernel: The way to carry out the interpolation.
            Options: 'x_linear': The velocity is assumed a constant and the trajectory will be linear.
                     'v_linear': The velocity is assumed linear.
                     'a_linear': The acceleration is assumed linear.

    '''

    ind = df.first_valid_index()
    df[:ind] = df[ind]
    x = df.values
    x_p = np.zeros_like(x)
    ind_not_non = np.argwhere(1 - np.isnan(x))
    if kernel == 'x_linear':
        for i, data in enumerate(x):
            if not np.isnan(data):
                x_p[i] = x[i]
            else:
                i_next = ind_not_non[(ind_not_non > i).argmax()]
                if not i_next == 0:
                    v = (x[i_next] - x_p[i - 1])/(i_next - i + 1)
                else:
                    v = x_p[i - 1] - x_p[i - 2]
                x_p[i] = x_p[i - 1] + v

    #TODO: kernel == 'v_linear', 'a_linear' and 'a_p_linear'
    return x_p


def remove_nans(df, remove_method = 'interpolation', kernel = 'x_linear', window_size = 11):
    '''
    This function takes the file(obtained from trangulation) that containes the trajectories of the markers as input.
    Return a dataframe without NaNs.

    file_path: path to the file that containes the 3D trajectories of the makers.

    nan_method: method to deal with the Nans in the dataframe. Options: 'drop', 'fill'
                'drop': This will just drop all the rows that contain Nans
                'fill': This will fill the Nans from previous non-Nan value if applicable. If there is no previous
                        non-Nan value, it will fill the Nans from the following non-Nan value.
                'ignore': Do nothing.
                'interpolation': Use interpolation to fill the NaNs.
    '''
    df_new = pd.DataFrame().reindex_like(df)


    if remove_method == 'fill':
        df = df.fillna(method = 'ffill').fillna(method = 'bfill') # get rid of NaNs
    elif remove_method == 'drop':
        df = df.dropna()
    elif remove_method == 'ignore':
        pass
    elif remove_method =='interpolation':
        for column in df.columns:
            data = df[column]
            df_new[column] = interpolation_for_nan(data, window_size, kernel)
        return df_new
    return df

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
